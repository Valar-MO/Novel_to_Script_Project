from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import httpx
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
)
from pydantic import ValidationError

from backend.config import (
    ConfigurationError,
    LLMSettings,
    get_llm_settings,
)
from backend.llm.base import (
    LLMCallMetadata,
    LLMHealthStatus,
    LLMModelNotFoundError,
    LLMProvider,
    LLMProviderUnavailableError,
    LLMRequestError,
    LLMResponseValidationError,
    ResponseModelT,
)


class CloudAPIProvider(LLMProvider):
    """OpenAI-compatible structured LLM provider."""

    def __init__(
        self,
        *,
        settings: LLMSettings | None = None,
        client: Any | None = None,
    ) -> None:
        self._settings = (
            settings
            if settings is not None
            else get_llm_settings()
        )

        if not self._settings.uses_cloud_api:
            raise ValueError(
                "CloudAPIProvider requires provider='cloud_api' or 'deepseek'."
            )

        if not self._settings.cloud_api_key:
            raise ConfigurationError(
                "Cloud API key is missing. Set LLM_API_KEY or DEEPSEEK_API_KEY."
            )

        if not self._settings.cloud_api_base_url:
            raise ConfigurationError(
                "Cloud API base URL is missing. Set LLM_API_BASE_URL."
            )

        if not self._settings.cloud_api_model:
            raise ConfigurationError(
                "Cloud API model is missing. Set LLM_API_MODEL or DEEPSEEK_MODEL."
            )

        if client is None:
            self._client = AsyncOpenAI(
                api_key=self._settings.cloud_api_key,
                base_url=self._settings.cloud_api_base_url,
                timeout=self._settings.cloud_api_timeout_seconds,
            )
            self._owns_client = True
        else:
            self._client = client
            self._owns_client = False

        self._closed = False

    @property
    def provider_name(self) -> str:
        if self._settings.provider == "deepseek":
            return "deepseek"
        return "cloud_api"

    @property
    def model_name(self) -> str:
        return self._settings.cloud_api_model or ""

    async def generate_structured(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        response_model: type[ResponseModelT],
        temperature: float | None = None,
        metadata: LLMCallMetadata | None = None,
    ) -> ResponseModelT:
        self._ensure_open()

        normalized_messages = self.validate_structured_request(
            messages=messages,
            response_model=response_model,
            temperature=temperature,
        )
        grounded_messages = self._attach_schema_instruction(
            normalized_messages,
            response_model=response_model,
        )
        request_temperature = (
            self._settings.temperature
            if temperature is None
            else temperature
        )

        response_content = await self._chat_content(
            messages=grounded_messages,
            temperature=request_temperature,
            metadata=metadata,
        )
        response_content = self._clean_json_content(response_content)

        if not response_content:
            raise LLMResponseValidationError(
                "Cloud API returned an empty response."
            )

        try:
            result = response_model.model_validate_json(
                response_content
            )
        except (
            ValidationError,
            ValueError,
            TypeError,
        ) as first_error:
            repaired_content = await self._repair_structured_response(
                response_content=response_content,
                validation_error=first_error,
                response_model=response_model,
                metadata=metadata,
            )

            try:
                result = response_model.model_validate_json(
                    repaired_content
                )
            except (
                ValidationError,
                ValueError,
                TypeError,
            ) as second_error:
                first_detail = self._format_validation_error_detail(
                    error=first_error,
                    response_content=response_content,
                )
                second_detail = self._format_validation_error_detail(
                    error=second_error,
                    response_content=repaired_content,
                )

                raise LLMResponseValidationError(
                    "Cloud API output and one repair attempt both failed "
                    f"{response_model.__name__} validation.\n\n"
                    f"First validation:\n{first_detail}\n\n"
                    f"After repair:\n{second_detail}"
                ) from second_error

        return cast(ResponseModelT, result)

    async def _repair_structured_response(
        self,
        *,
        response_content: str,
        validation_error: Exception,
        response_model: type[ResponseModelT],
        metadata: LLMCallMetadata | None = None,
    ) -> str:
        repair_instruction = (
            f"The previous output did not pass {response_model.__name__} "
            "validation. Return exactly one JSON object that directly matches "
            "the requested schema. Do not output Markdown fences, comments, "
            "explanations, wrapper objects, or translated field names. "
            "The first character must be { and the last character must be }. "
            "Drop records that cannot be repaired without inventing facts.\n\n"
            f"Validation error:\n{validation_error}\n\n"
            f"Original output:\n{response_content}"
        )

        repaired_content = await self._chat_content(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You repair JSON structures. Output only one valid "
                        "JSON object and nothing else."
                    ),
                },
                {
                    "role": "user",
                    "content": repair_instruction,
                },
            ],
            temperature=0,
            metadata=(
                LLMCallMetadata(
                    chunk_id=(metadata.chunk_id if metadata else None),
                    layer_name=(metadata.layer_name if metadata else None),
                    is_repair=True,
                )
            ),
        )
        repaired_content = self._clean_json_content(repaired_content)

        if not repaired_content:
            raise LLMResponseValidationError(
                "Cloud API repair returned an empty response."
            )

        return repaired_content

    async def _chat_content(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        metadata: LLMCallMetadata | None = None,
    ) -> str:
        del metadata

        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
            "response_format": {
                "type": "json_object",
            },
        }

        if self._settings.cloud_api_reasoning_effort:
            request_kwargs["reasoning_effort"] = (
                self._settings.cloud_api_reasoning_effort
            )

        if self._settings.cloud_api_thinking_enabled:
            request_kwargs["extra_body"] = {
                "thinking": {
                    "type": "enabled",
                }
            }

        try:
            response = await self._client.chat.completions.create(
                **request_kwargs
            )
        except AuthenticationError as error:
            raise LLMProviderUnavailableError(
                "Cloud API authentication failed. Check the API key."
            ) from error
        except NotFoundError as error:
            raise LLMModelNotFoundError(
                f"Cloud API model was not found: {self.model_name}"
            ) from error
        except BadRequestError as error:
            raise LLMRequestError(
                f"Cloud API rejected the request: {error}"
            ) from error
        except (
            APIConnectionError,
            APITimeoutError,
            httpx.TimeoutException,
            TimeoutError,
        ) as error:
            raise LLMProviderUnavailableError(
                "Cloud API request timed out or could not connect."
            ) from error
        except APIError as error:
            raise LLMRequestError(
                f"Cloud API request failed: {error}"
            ) from error
        except Exception as error:
            raise LLMRequestError(
                f"Unexpected Cloud API error: {error}"
            ) from error

        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMResponseValidationError(
                "Cloud API returned no choices."
            )

        message = choices[0].message
        content = getattr(message, "content", None)
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                else:
                    text = getattr(part, "text", None)
                if text:
                    text_parts.append(str(text))
            return "".join(text_parts)

        return str(content or "").strip()

    async def health_check(self) -> LLMHealthStatus:
        if self._closed:
            return LLMHealthStatus(
                provider=self.provider_name,
                model=self.model_name,
                available=False,
                detail="CloudAPIProvider is closed.",
            )

        if not self._settings.cloud_api_key:
            return LLMHealthStatus(
                provider=self.provider_name,
                model=self.model_name,
                available=False,
                detail="Cloud API key is missing.",
            )

        try:
            request_kwargs: dict[str, Any] = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": "Reply with OK.",
                    }
                ],
                "stream": False,
                "max_tokens": 16,
            }

            if self._settings.cloud_api_reasoning_effort:
                request_kwargs["reasoning_effort"] = (
                    self._settings.cloud_api_reasoning_effort
                )

            response = await self._client.chat.completions.create(
                **request_kwargs
            )
            choices = getattr(response, "choices", None) or []
            available = bool(choices)
            return LLMHealthStatus(
                provider=self.provider_name,
                model=self.model_name,
                available=available,
                detail=(
                    "Cloud API is reachable."
                    if available
                    else "Cloud API returned no choices."
                ),
            )
        except AuthenticationError:
            return LLMHealthStatus(
                provider=self.provider_name,
                model=self.model_name,
                available=False,
                detail="Cloud API authentication failed.",
            )
        except NotFoundError:
            return LLMHealthStatus(
                provider=self.provider_name,
                model=self.model_name,
                available=False,
                detail=f"Cloud API model was not found: {self.model_name}",
            )
        except Exception as error:
            return LLMHealthStatus(
                provider=self.provider_name,
                model=self.model_name,
                available=False,
                detail=f"Cloud API health check failed: {error}",
            )

    async def close(self) -> None:
        if self._closed:
            return

        self._closed = True

        if self._owns_client:
            await self._client.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise LLMProviderUnavailableError(
                "CloudAPIProvider is closed."
            )

    @staticmethod
    def _attach_schema_instruction(
        messages: list[dict[str, str]],
        *,
        response_model: type[ResponseModelT],
    ) -> list[dict[str, str]]:
        schema_json = response_model.model_json_schema()
        instruction = (
            "Return only one valid JSON object. Do not output Markdown, "
            "comments, explanations, code fences, or extra text. Field names "
            "must exactly match this JSON Schema. Do not add wrapper fields.\n\n"
            f"JSON Schema:\n{schema_json}"
        )

        grounded_messages = [dict(message) for message in messages]

        for message in grounded_messages:
            if message["role"] == "system":
                message["content"] = (
                    f"{message['content']}\n\n{instruction}"
                )
                return grounded_messages

        grounded_messages.insert(
            0,
            {
                "role": "system",
                "content": instruction,
            },
        )
        return grounded_messages

    @staticmethod
    def _clean_json_content(content: str) -> str:
        cleaned = content.strip()

        if cleaned.startswith("```"):
            lines = cleaned.splitlines()

            if lines and lines[0].strip().lower() in {
                "```",
                "```json",
                "```javascript",
            }:
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            cleaned = "\n".join(lines).strip()

        return CloudAPIProvider._extract_first_json_object(cleaned)

    @staticmethod
    def _extract_first_json_object(content: str) -> str:
        start_index = content.find("{")

        if start_index < 0:
            return content.strip()

        depth = 0
        in_string = False
        escaped = False

        for index in range(start_index, len(content)):
            character = content[index]

            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == "\"":
                    in_string = False
                continue

            if character == "\"":
                in_string = True
                continue

            if character == "{":
                depth += 1
                continue

            if character == "}":
                depth -= 1

                if depth == 0:
                    return content[start_index:index + 1].strip()

        return content.strip()

    @staticmethod
    def _format_validation_error_detail(
        *,
        error: Exception,
        response_content: str,
    ) -> str:
        raw_preview = response_content[:800]

        if isinstance(error, ValidationError):
            messages: list[str] = []
            errors = error.errors()

            for item in errors[:12]:
                location = ".".join(
                    str(part)
                    for part in item.get("loc", ())
                )
                message = item.get("msg", "validation error")
                if location:
                    messages.append(f"field {location}: {message}")
                else:
                    messages.append(str(message))

            if len(errors) > 12:
                messages.append(
                    f"{len(errors) - 12} more validation errors omitted."
                )

            validation_detail = "\n".join(messages)
        else:
            validation_detail = str(error)

        return (
            "Validation error:\n"
            f"{validation_detail}\n\n"
            "Raw model output preview:\n"
            f"{raw_preview}"
        )
