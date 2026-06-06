
from __future__ import annotations

from collections.abc import Mapping

from backend.config import (
    ConfigurationError,
    LLMSettings,
    get_llm_settings,
)
from backend.llm.base import LLMProvider
from backend.llm.mock_provider import (
    MockPayload,
    MockProvider,
)
from backend.llm.ollama_provider import (
    OllamaProvider,
)


def create_llm_provider(
    *,
    settings: LLMSettings | None = None,
    mock_responses: Mapping[
        str,
        MockPayload,
    ]
    | None = None,
) -> LLMProvider:
    """
    根据应用配置创建对应的大模型 Provider。

    当前支持：
    - ollama：本地 Ollama 模型
    - mock：固定测试结果

    cloud_api 只预留配置，尚未实现。
    """

    resolved_settings = (
        settings
        if settings is not None
        else get_llm_settings()
    )

    if resolved_settings.provider == "ollama":
        return OllamaProvider(
            settings=resolved_settings
        )

    if resolved_settings.provider == "mock":
        return MockProvider(
            responses=mock_responses,
            model_name="mock-character-model",
        )

    if (
        resolved_settings.provider
        == "cloud_api"
    ):
        raise ConfigurationError(
            "LLM_PROVIDER 已设置为 cloud_api，"
            "但 CloudAPIProvider 尚未实现。"
            "当前请使用 ollama 或 mock。"
        )

    # 理论上 config.py 已经拦截未知值，
    # 这里保留防御性检查。
    raise ConfigurationError(
        "无法创建未知的大模型 Provider："
        f"{resolved_settings.provider!r}"
    )

