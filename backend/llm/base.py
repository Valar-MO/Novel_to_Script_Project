
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Literal,
    Mapping,
    Sequence,
    TypeVar,
)

from pydantic import BaseModel


LLMRole = Literal[
    "system",
    "user",
    "assistant",
]

ResponseModelT = TypeVar(
    "ResponseModelT",
    bound=BaseModel,
)


class LLMProviderError(RuntimeError):
    """大模型提供方调用过程中发生的基础异常。"""


class LLMProviderUnavailableError(
    LLMProviderError
):
    """模型服务当前不可用。"""


class LLMModelNotFoundError(
    LLMProviderError
):
    """配置的模型不存在或尚未安装。"""


class LLMRequestError(
    LLMProviderError
):
    """模型请求参数不合法或请求执行失败。"""


class LLMResponseValidationError(
    LLMProviderError
):
    """模型输出无法通过目标 Pydantic 模型校验。"""


@dataclass(frozen=True, slots=True)
class LLMCallMetadata:
    """附加到模型调用上的上下文信息。"""

    chunk_id: str | None = None
    layer_name: str | None = None
    is_repair: bool = False


@dataclass(
    frozen=True,
    slots=True,
)
class LLMHealthStatus:
    """模型提供方健康检查结果。"""

    provider: str
    model: str
    available: bool
    detail: str

    def to_dict(
        self,
    ) -> dict[str, str | bool]:
        """转换为可序列化字典。"""

        return {
            "provider": self.provider,
            "model": self.model,
            "available": self.available,
            "detail": self.detail,
        }


class LLMProvider(ABC):
    """
    Novel2Script 的统一大模型提供方接口。

    人物分析、事件分析和场景划分等业务模块只依赖
    这个接口，不直接依赖具体云端 SDK。
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """提供方名称，例如 deepseek 或 mock。"""

        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """当前使用的模型名称。"""

        raise NotImplementedError

    @abstractmethod
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
        调用模型并返回经过 Pydantic 校验的结构化结果。

        Parameters
        ----------
        messages:
            模型对话消息，每条消息至少包含 role 和 content。

        response_model:
            预期响应对应的 Pydantic 模型类型。

        temperature:
            本次请求单独使用的温度参数。
            为 None 时由具体 Provider 使用默认配置。
        """

        raise NotImplementedError

    @abstractmethod
    async def health_check(
        self,
    ) -> LLMHealthStatus:
        """检查模型服务和模型是否可用。"""

        raise NotImplementedError

    async def close(
        self,
    ) -> None:
        """
        释放 Provider 使用的资源。

        当前 Mock 可能不需要显式关闭，
        云端客户端以后可以覆盖该方法。
        """

        return None

    async def __aenter__(
        self,
    ) -> "LLMProvider":
        return self

    async def __aexit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        await self.close()

    @staticmethod
    def normalize_messages(
        messages: Sequence[
            Mapping[str, str]
        ],
    ) -> list[dict[str, str]]:
        """
        校验并规范化模型消息。

        返回普通字典列表，方便直接交给不同模型 SDK。
        """

        if not messages:
            raise LLMRequestError(
                "模型消息列表不能为空。"
            )

        supported_roles = {
            "system",
            "user",
            "assistant",
        }

        normalized_messages: list[
            dict[str, str]
        ] = []

        for index, message in enumerate(
            messages,
            start=1,
        ):
            role = str(
                message.get("role", "")
            ).strip()

            content = str(
                message.get("content", "")
            ).strip()

            if role not in supported_roles:
                raise LLMRequestError(
                    f"第 {index} 条消息的 role "
                    f"不合法：{role!r}。"
                )

            if not content:
                raise LLMRequestError(
                    f"第 {index} 条消息的 "
                    "content 不能为空。"
                )

            normalized_messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        return normalized_messages

    @staticmethod
    def validate_response_model(
        response_model: type[ResponseModelT],
    ) -> None:
        """确认目标响应类型是 Pydantic 模型。"""

        try:
            is_pydantic_model = issubclass(
                response_model,
                BaseModel,
            )
        except TypeError as error:
            raise LLMRequestError(
                "response_model 必须是 "
                "Pydantic BaseModel 子类。"
            ) from error

        if not is_pydantic_model:
            raise LLMRequestError(
                "response_model 必须是 "
                "Pydantic BaseModel 子类。"
            )

    @staticmethod
    def validate_temperature(
        temperature: float | None,
    ) -> None:
        """校验可选的模型温度参数。"""

        if temperature is None:
            return

        if not 0 <= temperature <= 2:
            raise LLMRequestError(
                "temperature 必须位于 "
                "0 到 2 之间。"
            )

    @classmethod
    def validate_structured_request(
        cls,
        *,
        messages: Sequence[
            Mapping[str, str]
        ],
        response_model: type[ResponseModelT],
        temperature: float | None,
    ) -> list[dict[str, str]]:
        """统一校验一次结构化模型请求。"""

        cls.validate_response_model(
            response_model
        )

        cls.validate_temperature(
            temperature
        )

        return cls.normalize_messages(
            messages
        )
