
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
    - mock：固定测试结果
    - deepseek/cloud_api：OpenAI-compatible API

    运行时只创建 mock 或 OpenAI-compatible API provider。
    """

    resolved_settings = (
        settings
        if settings is not None
        else get_llm_settings()
    )

    if resolved_settings.provider == "mock":
        return MockProvider(
            responses=mock_responses,
            model_name="mock-character-model",
        )

    if resolved_settings.uses_cloud_api:
        from backend.llm.cloud_api_provider import (
            CloudAPIProvider,
        )

        return CloudAPIProvider(
            settings=resolved_settings
        )

    # 理论上 config.py 已经拦截未知值，
    # 这里保留防御性检查。
    raise ConfigurationError(
        "无法创建未知的大模型 Provider："
        f"{resolved_settings.provider!r}"
    )

