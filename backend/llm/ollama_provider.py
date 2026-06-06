
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import httpx
from ollama import (
    AsyncClient,
    RequestError,
    ResponseError,
)
from pydantic import BaseModel, ValidationError

from backend.config import (
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


class OllamaProvider(LLMProvider):
    """通过本地 Ollama 服务调用结构化大模型。"""

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

        if self._settings.provider != "ollama":
            raise ValueError(
                "OllamaProvider 只能使用 "
                "provider='ollama' 的配置。"
            )

        if client is None:
            self._client = AsyncClient(
                host=self._settings.ollama_base_url,
                timeout=(
                    self._settings
                    .ollama_timeout_seconds
                ),
            )
            self._owns_client = True
        else:
            # 测试时可以注入 AsyncMock，避免调用真实模型。
            self._client = client
            self._owns_client = False

        self._closed = False

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._settings.ollama_model

    async def generate_structured(
        self,
        *,
        messages: Sequence[
            Mapping[str, str]
        ],
        response_model: type[ResponseModelT],
        temperature: float | None = None,
        metadata: LLMCallMetadata | None = None,
    ) -> ResponseModelT:
        """
        调用 Ollama，并将返回内容校验成目标 Pydantic 模型。
        """

        self._ensure_open()

        normalized_messages = (
            self.validate_structured_request(
                messages=messages,
                response_model=response_model,
                temperature=temperature,
            )
        )

        grounded_messages = (
            self._attach_schema_instruction(
                normalized_messages,
                response_model,
            )
        )

        json_schema = (
            response_model.model_json_schema()
        )
        request_temperature = (
            self._settings.temperature
            if temperature is None
            else temperature
        )

        response = await self._chat_with_error_conversion(
            messages=grounded_messages,
            json_schema=json_schema,
            temperature=request_temperature,
            metadata=metadata,
        )

        response_content = (
            self._extract_response_content(response)
        )
        response_content = (
            self._clean_json_content(response_content)
        )

        if not response_content:
            raise LLMResponseValidationError(
                "Ollama 返回了空响应，"
                "无法进行结构化结果校验。"
            )

        try:
            result = (
                response_model.model_validate_json(
                    response_content
                )
            )
        except (
            ValidationError,
            ValueError,
            TypeError,
        ) as first_error:
            repaired_content = (
                await self._repair_structured_response(
                    grounded_messages=grounded_messages,
                    response_content=response_content,
                    validation_error=first_error,
                    json_schema=json_schema,
                    response_model_name=(
                        response_model.__name__
                    ),
                    metadata=metadata,
                )
            )

            try:
                result = (
                    response_model.model_validate_json(
                        repaired_content
                    )
                )
            except (
                ValidationError,
                ValueError,
                TypeError,
            ) as second_error:
                first_detail = (
                    self._format_validation_error_detail(
                        error=first_error,
                        response_content=response_content,
                    )
                )
                second_detail = (
                    self._format_validation_error_detail(
                        error=second_error,
                        response_content=repaired_content,
                    )
                )

                raise LLMResponseValidationError(
                    "Ollama 原始输出和一次自动修复后的输出，"
                    f"均无法通过 {response_model.__name__} "
                    "结构校验。\n\n"
                    f"第一次校验：\n{first_detail}\n\n"
                    f"修复后校验：\n{second_detail}"
                ) from second_error

        return cast(
            ResponseModelT,
            result,
        )

    async def _chat_with_error_conversion(
        self,
        *,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any],
        temperature: float,
        metadata: LLMCallMetadata | None = None,
    ) -> Any:
        """调用 Ollama，并转换 SDK/网络异常为项目统一异常。"""

        try:
            return await self._client.chat(
                model=self.model_name,
                messages=messages,
                stream=False,
                think=False,
                format=json_schema,
                options={
                    "temperature": temperature,
                    "num_predict": 2048,
                },
                keep_alive=(
                    self._settings.keep_alive
                ),
            )

        except ResponseError as error:
            self._raise_response_error(error)

        except RequestError as error:
            raise LLMRequestError(
                "Ollama 拒绝了模型请求："
                f"{error}"
            ) from error

        except (
            ConnectionError,
            httpx.ConnectError,
        ) as error:
            raise LLMProviderUnavailableError(
                "无法连接本地 Ollama 服务，"
                "请确认 Ollama 已经启动，且地址为："
                f"{self._settings.ollama_base_url}"
            ) from error

        except (
            httpx.TimeoutException,
            TimeoutError,
        ) as error:
            stage_label = "自动修复阶段" if metadata and metadata.is_repair else "主调用阶段"
            chunk_label = metadata.chunk_id if metadata and metadata.chunk_id else "unknown_chunk"
            layer_label = metadata.layer_name if metadata and metadata.layer_name else "unknown_layer"
            raise LLMProviderUnavailableError(
                f"{chunk_label} 的 {layer_label} 层在 "
                f"{self._settings.ollama_timeout_seconds:g} 秒后超时"
                f"（{stage_label}）。"
            ) from error

        except LLMProviderUnavailableError:
            raise

        except Exception as error:
            raise LLMRequestError(
                "调用 Ollama 时发生未预期错误："
                f"{error}"
            ) from error

    async def _repair_structured_response(
        self,
        *,
        grounded_messages: list[dict[str, str]],
        response_content: str,
        validation_error: Exception,
        json_schema: dict[str, Any],
        response_model_name: str,
        metadata: LLMCallMetadata | None = None,
    ) -> str:
        repair_instruction = (
            f"上一份输出没有通过 {response_model_name} 校验。"
            f"请返回一个最外层直接符合 {response_model_name} "
            "的 JSON 对象。"
            "字段必须严格遵守当前结构要求，"
            "不得增加包装层、type、extraction、chapter 等额外字段。"
            "不得输出 ```json、```、Markdown、解释或其他文字。"
            "输出的第一个字符必须是 {，最后一个字符必须是 }。"
            "无法确认的集合返回空数组。"
            "请只修复 JSON 结构，不得编造原文不存在的事实。"
            "mention 记录必须使用 mention_type、mention_text、"
            "evidence_text、confidence。"
            "relation 记录必须使用 source_mention、relation、"
            "target_mention、evidence_text、confidence。"
            "event frame 记录必须使用 trigger_text、event_type、"
            "arguments、evidence_text、confidence。"
            "event_type 只能使用 movement、communication、perception、"
            "cognition、state、possession、social、creation、conflict、other；"
            "无法确定时使用 other。"
            "删除 background、identity、biography、analysis、summary "
            "等非法字段。"
            "warnings 只能出现在最外层。"
            "每条记录必须包含合法的 evidence_text 和 "
            "0 到 1 之间的 confidence。"
            "无法补全必填字段的记录必须整条删除。"
            "event frame 的 arguments 必须是 role、mention_id、mention_text 对象数组；"
            "mention_id 必须保留原输出或输入中的 mention_id，不得删除。"
            "\n\n"
            f"校验错误：\n{validation_error}\n\n"
            f"原始输出：\n{response_content}"
        )

        repair_response = (
            await self._chat_with_error_conversion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是 JSON 结构修复器。"
                            "只输出一个合法 JSON 对象。"
                            "不得输出 Markdown 代码围栏、解释或其他文本。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": repair_instruction,
                    },
                ],
                json_schema=json_schema,
                temperature=0,
                metadata=(
                    LLMCallMetadata(
                        chunk_id=(metadata.chunk_id if metadata else None),
                        layer_name=(metadata.layer_name if metadata else None),
                        is_repair=True,
                    )
                ),
            )
        )

        repaired_content = (
            self._extract_response_content(
                repair_response
            )
        )
        repaired_content = (
            self._clean_json_content(repaired_content)
        )

        if not repaired_content:
            raise LLMResponseValidationError(
                "Ollama 自动修复返回了空内容。"
            )

        return repaired_content

    @staticmethod
    def _clean_json_content(
        content: str,
    ) -> str:
        """清除围栏，并提取输出中的第一个完整 JSON 对象。"""

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

        return OllamaProvider._extract_first_json_object(
            cleaned
        )

    @staticmethod
    def _extract_first_json_object(
        content: str,
    ) -> str:
        """
        Return the first balanced JSON object in content.

        Ollama occasionally appends explanations after a valid object. Pydantic
        rejects that as trailing characters, so we conservatively keep only
        the first balanced object while respecting strings and escapes.
        """

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
                    return content[
                        start_index:index + 1
                    ].strip()

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
                message = item.get(
                    "msg",
                    "validation error",
                )

                if location:
                    messages.append(
                        f"字段 {location}：{message}"
                    )
                else:
                    messages.append(str(message))

            if len(errors) > 12:
                messages.append(
                    f"另有 {len(errors) - 12} 个校验错误未显示。"
                )

            validation_detail = "\n".join(messages)
        else:
            validation_detail = str(error)

        return (
            "结构校验错误：\n"
            f"{validation_detail}\n\n"
            "模型原始输出前 800 字：\n"
            f"{raw_preview}"
        )

    async def health_check(
        self,
    ) -> LLMHealthStatus:
        """
        检查 Ollama 服务是否可访问，以及配置模型是否已安装。
        """

        if self._closed:
            return LLMHealthStatus(
                provider=self.provider_name,
                model=self.model_name,
                available=False,
                detail=(
                    "OllamaProvider 已关闭。"
                ),
            )

        try:
            response = await self._client.list()

            installed_models = (
                self._extract_installed_model_names(
                    response
                )
            )

            if self._model_is_installed(
                configured_model=self.model_name,
                installed_models=installed_models,
            ):
                return LLMHealthStatus(
                    provider=self.provider_name,
                    model=self.model_name,
                    available=True,
                    detail=(
                        "Ollama 服务可用，"
                        f"模型 {self.model_name} "
                        "已经安装。"
                    ),
                )

            model_list = (
                "、".join(
                    sorted(installed_models)
                )
                if installed_models
                else "无"
            )

            return LLMHealthStatus(
                provider=self.provider_name,
                model=self.model_name,
                available=False,
                detail=(
                    "Ollama 服务可以访问，"
                    f"但没有找到模型 {self.model_name}。"
                    f"当前模型：{model_list}。"
                ),
            )

        except (
            ConnectionError,
            httpx.ConnectError,
        ):
            return LLMHealthStatus(
                provider=self.provider_name,
                model=self.model_name,
                available=False,
                detail=(
                    "无法连接本地 Ollama 服务，"
                    "请确认 Ollama 已经启动。"
                ),
            )

        except (
            httpx.TimeoutException,
            TimeoutError,
        ):
            return LLMHealthStatus(
                provider=self.provider_name,
                model=self.model_name,
                available=False,
                detail="Ollama 健康检查超时。",
            )

        except ResponseError as error:
            return LLMHealthStatus(
                provider=self.provider_name,
                model=self.model_name,
                available=False,
                detail=(
                    "Ollama 健康检查失败："
                    f"{error}"
                ),
            )

        except Exception as error:
            return LLMHealthStatus(
                provider=self.provider_name,
                model=self.model_name,
                available=False,
                detail=(
                    "Ollama 健康检查发生"
                    "未预期错误："
                    f"{error}"
                ),
            )

    async def close(
        self,
    ) -> None:
        """关闭由当前 Provider 创建的异步客户端。"""

        if self._closed:
            return

        self._closed = True

        if self._owns_client:
            await self._client.close()

    def _ensure_open(
        self,
    ) -> None:
        if self._closed:
            raise LLMProviderUnavailableError(
                "OllamaProvider 已经关闭，"
                "无法继续发送请求。"
            )

    @staticmethod
    def _attach_schema_instruction(
        messages: list[dict[str, str]],
        response_model: type[BaseModel],
    ) -> list[dict[str, str]]:
        """
        Add a short structured-output instruction without inlining the full schema.

        Ollama already receives the formal schema via format=json_schema.
        Repeating the entire schema in prompts increases token cost and latency.
        """

        schema_instruction = (
            "你必须只返回一个合法 JSON 对象。"
            "不得输出 Markdown、解释、代码围栏或额外文本。"
            "所有 JSON 字段名必须严格使用预期结构中的英文名称，"
            "不得翻译字段名，不得自行增加字段或包装层。"
            "数组、null、字符串、数字和布尔值必须符合预期结构要求。"
        )

        grounded_messages = [
            dict(message)
            for message in messages
        ]

        for message in grounded_messages:
            if message["role"] == "system":
                message["content"] = (
                    f"{message['content']}\n\n"
                    f"{schema_instruction}"
                )
                return grounded_messages

        grounded_messages.insert(
            0,
            {
                "role": "system",
                "content": schema_instruction,
            },
        )

        return grounded_messages

    @staticmethod
    def _extract_response_content(
        response: Any,
    ) -> str:
        """从 Ollama ChatResponse 中提取最终文本。"""

        message = getattr(
            response,
            "message",
            None,
        )

        if message is None and isinstance(
            response,
            Mapping,
        ):
            message = response.get("message")

        if message is None:
            return ""

        content = getattr(
            message,
            "content",
            None,
        )

        if content is None and isinstance(
            message,
            Mapping,
        ):
            content = message.get("content")

        if content is None:
            return ""

        return str(content).strip()

    @staticmethod
    def _extract_installed_model_names(
        response: Any,
    ) -> set[str]:
        """从 Ollama 模型列表响应中读取模型名称。"""

        models = getattr(
            response,
            "models",
            None,
        )

        if models is None and isinstance(
            response,
            Mapping,
        ):
            models = response.get("models")

        if not models:
            return set()

        model_names: set[str] = set()

        for model_item in models:
            model_name = getattr(
                model_item,
                "model",
                None,
            )

            if (
                model_name is None
                and isinstance(
                    model_item,
                    Mapping,
                )
            ):
                model_name = (
                    model_item.get("model")
                    or model_item.get("name")
                )

            if model_name:
                model_names.add(
                    str(model_name).strip()
                )

        return model_names

    @staticmethod
    def _model_is_installed(
        *,
        configured_model: str,
        installed_models: set[str],
    ) -> bool:
        """判断配置模型是否存在于 Ollama 模型列表中。"""

        normalized_configured = (
            configured_model.casefold()
        )

        normalized_installed = {
            model_name.casefold()
            for model_name in installed_models
        }

        if (
            normalized_configured
            in normalized_installed
        ):
            return True

        # 当配置为 qwen3 时，允许匹配 qwen3:latest。
        if ":" not in normalized_configured:
            return any(
                installed_name.split(
                    ":",
                    maxsplit=1,
                )[0]
                == normalized_configured
                for installed_name
                in normalized_installed
            )

        return False

    def _raise_response_error(
        self,
        error: ResponseError,
    ) -> None:
        """把 Ollama SDK 异常转换为项目统一异常。"""

        status_code = getattr(
            error,
            "status_code",
            -1,
        )

        error_message = getattr(
            error,
            "error",
            str(error),
        )

        if status_code == 404:
            raise LLMModelNotFoundError(
                f"Ollama 中不存在模型 "
                f"{self.model_name}。"
                "请先执行："
                f"ollama pull {self.model_name}"
            ) from error

        if status_code in {
            500,
            502,
            503,
            504,
        }:
            raise LLMProviderUnavailableError(
                "Ollama 服务暂时不可用："
                f"{error_message}"
            ) from error

        raise LLMRequestError(
            "Ollama 请求失败："
            f"{error_message}"
        ) from error

