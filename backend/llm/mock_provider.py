
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import (
    Any,
    Mapping,
    Sequence,
    cast,
)

from pydantic import (
    BaseModel,
    ValidationError,
)

from backend.llm.base import (
    LLMHealthStatus,
    LLMProvider,
    LLMProviderUnavailableError,
    LLMRequestError,
    LLMResponseValidationError,
    ResponseModelT,
)
from backend.llm.schemas import (
    CharacterCandidateExtractionOutput,
    EventFrameExtractionOutput,
    MentionExtractionOutput,
    RelationExtractionOutput,
    ScriptGenerationOutput,
)


MockPayload = (
    BaseModel
    | Mapping[str, Any]
    | str
)


class MockResponseNotConfiguredError(
    LLMRequestError
):
    """Mock Provider 没有配置目标响应。"""


@dataclass(
    frozen=True,
    slots=True,
)
class MockLLMCall:
    """一次 Mock 模型调用记录。"""

    messages: tuple[
        tuple[str, str],
        ...,
    ]

    response_model_name: str
    temperature: float | None


class MockProvider(LLMProvider):
    """
    用于自动测试和无模型演示的 LLM Provider。

    它不会访问网络，也不会调用 Ollama，而是返回预先配置
    的 Pydantic 模型、字典或 JSON 字符串。

    支持为同一种响应模型排队多个结果，便于模拟不同文本块
    分别返回不同人物提取结果。
    """

    def __init__(
        self,
        *,
        responses: Mapping[
            str,
            MockPayload,
        ]
        | None = None,
        available: bool = True,
        model_name: str = "mock-model",
    ) -> None:
        self._available = available
        self._model_name = (
            model_name.strip()
            or "mock-model"
        )

        self._response_queues: dict[
            str,
            deque[MockPayload],
        ] = defaultdict(deque)

        self._calls: list[
            MockLLMCall
        ] = []

        if responses:
            for (
                response_model_name,
                payload,
            ) in responses.items():
                self._response_queues[
                    response_model_name
                ].append(payload)

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def calls(
        self,
    ) -> tuple[MockLLMCall, ...]:
        """
        返回调用记录的只读快照。

        单元测试可用它检查消息内容、响应模型和 temperature。
        """

        return tuple(self._calls)

    @property
    def call_count(self) -> int:
        """返回当前累计调用次数。"""

        return len(self._calls)

    def set_available(
        self,
        available: bool,
    ) -> None:
        """设置 Mock 服务是否可用。"""

        self._available = available

    def register_response(
        self,
        response_model: type[BaseModel],
        payload: MockPayload,
        *,
        clear_existing: bool = False,
    ) -> None:
        """
        为指定响应模型注册一个返回结果。

        多次调用会按照注册顺序依次返回。
        """

        self.validate_response_model(
            response_model
        )

        response_model_name = (
            response_model.__name__
        )

        if clear_existing:
            self._response_queues[
                response_model_name
            ].clear()

        self._response_queues[
            response_model_name
        ].append(payload)

    def clear_responses(
        self,
    ) -> None:
        """清除所有已注册的 Mock 响应。"""

        self._response_queues.clear()

    def clear_calls(
        self,
    ) -> None:
        """清除模型调用记录。"""

        self._calls.clear()

    async def generate_structured(
        self,
        *,
        messages: Sequence[
            Mapping[str, str]
        ],
        response_model: type[ResponseModelT],
        temperature: float | None = None,
        metadata: object | None = None,
    ) -> ResponseModelT:
        """
        返回预先配置且经过 Pydantic 校验的结构化结果。
        """

        normalized_messages = (
            self.validate_structured_request(
                messages=messages,
                response_model=response_model,
                temperature=temperature,
            )
        )

        if not self._available:
            raise LLMProviderUnavailableError(
                "Mock LLM Provider 当前不可用。"
            )

        self._calls.append(
            MockLLMCall(
                messages=tuple(
                    (
                        message["role"],
                        message["content"],
                    )
                    for message
                    in normalized_messages
                ),
                response_model_name=(
                    response_model.__name__
                ),
                temperature=temperature,
            )
        )

        payload = self._take_payload(
            response_model
        )

        return self._validate_payload(
            payload=payload,
            response_model=response_model,
        )

    async def health_check(
        self,
    ) -> LLMHealthStatus:
        """返回 Mock Provider 的模拟健康状态。"""

        if self._available:
            return LLMHealthStatus(
                provider=self.provider_name,
                model=self.model_name,
                available=True,
                detail=(
                    "Mock LLM Provider 可用，"
                    "不会调用真实模型。"
                ),
            )

        return LLMHealthStatus(
            provider=self.provider_name,
            model=self.model_name,
            available=False,
            detail=(
                "Mock LLM Provider "
                "已被设置为不可用。"
            ),
        )

    def _take_payload(
        self,
        response_model: type[ResponseModelT],
    ) -> MockPayload:
        """
        取得目标模型的下一条 Mock 响应。

        MentionExtractionOutput, RelationExtractionOutput and
        EventFrameExtractionOutput can return
        empty outputs by default, which keeps no-fact chunks easy to test.
        """

        response_model_name = (
            response_model.__name__
        )

        response_queue = self._response_queues.get(
            response_model_name
        )

        if response_queue:
            return response_queue.popleft()

        if (
            response_model
            is MentionExtractionOutput
        ):
            return MentionExtractionOutput()

        if (
            response_model
            is RelationExtractionOutput
        ):
            return RelationExtractionOutput()

        if (
            response_model
            is EventFrameExtractionOutput
        ):
            return EventFrameExtractionOutput()

        if (
            response_model
            is CharacterCandidateExtractionOutput
        ):
            return CharacterCandidateExtractionOutput()

        if (
            response_model
            is ScriptGenerationOutput
        ):
            return ScriptGenerationOutput()

        raise MockResponseNotConfiguredError(
            "没有为响应模型 "
            f"{response_model_name} "
            "配置 Mock 返回结果。"
        )

    @staticmethod
    def _validate_payload(
        *,
        payload: MockPayload,
        response_model: type[ResponseModelT],
    ) -> ResponseModelT:
        """将 Mock 数据转换并校验为目标 Pydantic 模型。"""

        try:
            if isinstance(
                payload,
                response_model,
            ):
                # 返回深拷贝，避免业务代码修改后影响下一次测试。
                return cast(
                    ResponseModelT,
                    payload.model_copy(
                        deep=True
                    ),
                )

            if isinstance(
                payload,
                BaseModel,
            ):
                validated_result = (
                    response_model.model_validate(
                        payload.model_dump()
                    )
                )

                return cast(
                    ResponseModelT,
                    validated_result,
                )

            if isinstance(
                payload,
                str,
            ):
                validated_result = (
                    response_model
                    .model_validate_json(
                        payload
                    )
                )

                return cast(
                    ResponseModelT,
                    validated_result,
                )

            validated_result = (
                response_model.model_validate(
                    dict(payload)
                )
            )

            return cast(
                ResponseModelT,
                validated_result,
            )

        except (
            ValidationError,
            TypeError,
            ValueError,
        ) as error:
            raise LLMResponseValidationError(
                "Mock 返回结果无法通过 "
                f"{response_model.__name__} "
                "结构校验。"
            ) from error

