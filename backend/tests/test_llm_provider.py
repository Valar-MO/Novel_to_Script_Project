
import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

from backend.config import (
    ConfigurationError,
    LLMSettings,
)
from backend.llm.base import (
    LLMProviderUnavailableError,
    LLMRequestError,
    LLMResponseValidationError,
)
from backend.llm.factory import (
    create_llm_provider,
)
from backend.llm.mock_provider import (
    MockProvider,
)
from backend.llm.schemas import (
    MentionExtractionOutput,
)

try:
    from backend.llm.cloud_api_provider import (
        CloudAPIProvider,
    )
except ModuleNotFoundError:
    CloudAPIProvider = None  # type: ignore[assignment]


def build_test_settings(
    *,
    provider: str = "deepseek",
) -> LLMSettings:
    """创建不依赖本地 .env 的测试配置。"""

    return LLMSettings(
        provider=provider,  # type: ignore[arg-type]
        temperature=0.0,
        cloud_api_base_url=None,
        cloud_api_key=None,
        cloud_api_model=None,
        cloud_api_timeout_seconds=30.0,
        cloud_api_reasoning_effort=None,
        cloud_api_thinking_enabled=False,
    )


def build_mention_payload() -> dict:
    """构造一份合法文本锚点抽取结果。"""

    return {
        "mentions": [
            {
                "mention_type": "character",
                "mention_text": "韩立",
                "evidence_text": "韩立走进山谷",
                "confidence": 0.95,
            }
        ]
    }


class TestMockProvider(
    unittest.IsolatedAsyncioTestCase
):
    """测试不会调用真实模型的 Mock Provider。"""

    async def test_returns_registered_structured_result(
        self,
    ):
        provider = MockProvider()

        provider.register_response(
            MentionExtractionOutput,
            build_mention_payload(),
        )

        result = await provider.generate_structured(
            messages=[
                {
                    "role": "user",
                    "content": "提取人物。",
                }
            ],
            response_model=(
                MentionExtractionOutput
            ),
            temperature=0,
        )

        self.assertEqual(len(result.mentions), 1)
        self.assertEqual(result.mentions[0].mention_text, "韩立")

        self.assertEqual(
            provider.call_count,
            1,
        )

        call = provider.calls[0]

        self.assertEqual(
            call.response_model_name,
            "MentionExtractionOutput",
        )
        self.assertEqual(
            call.temperature,
            0,
        )

    async def test_returns_empty_mention_list_by_default(
        self,
    ):
        provider = MockProvider()

        result = await provider.generate_structured(
            messages=[
                {
                    "role": "user",
                    "content": "这段文字中没有人物。",
                }
            ],
            response_model=(
                MentionExtractionOutput
            ),
        )

        self.assertEqual(result.mentions, [])

    async def test_unavailable_mock_provider_raises_error(
        self,
    ):
        provider = MockProvider(
            available=False
        )

        with self.assertRaises(
            LLMProviderUnavailableError
        ):
            await provider.generate_structured(
                messages=[
                    {
                        "role": "user",
                        "content": "测试。",
                    }
                ],
                response_model=(
                    MentionExtractionOutput
                ),
            )

    async def test_invalid_messages_raise_request_error(
        self,
    ):
        provider = MockProvider()

        with self.assertRaises(
            LLMRequestError
        ):
            await provider.generate_structured(
                messages=[],
                response_model=(
                    MentionExtractionOutput
                ),
            )

    async def test_invalid_payload_raises_validation_error(
        self,
    ):
        provider = MockProvider()

        provider.register_response(
            MentionExtractionOutput,
            {
                "mentions": [
                    {
                        "mention_type": "character",
                        "mention_text": "",
                        "evidence_text": "",
                        "confidence": 3,
                    }
                ]
            },
        )

        with self.assertRaises(
            LLMResponseValidationError
        ):
            await provider.generate_structured(
                messages=[
                    {
                        "role": "user",
                        "content": "测试。",
                    }
                ],
                response_model=(
                    MentionExtractionOutput
                ),
            )



class TestLLMProviderFactory(
    unittest.IsolatedAsyncioTestCase
):
    """测试根据配置创建不同 Provider。"""

    async def test_factory_creates_mock_provider(
        self,
    ):
        settings = build_test_settings(
            provider="mock"
        )

        provider = create_llm_provider(
            settings=settings,
            mock_responses={
                "MentionExtractionOutput": {
                    "mentions": [],
                },
            },
        )

        self.assertIsInstance(
            provider,
            MockProvider,
        )

        result = await (
            provider.generate_structured(
                messages=[
                    {
                        "role": "user",
                        "content": "测试。",
                    }
                ],
                response_model=(
                    MentionExtractionOutput
                ),
            )
        )

        self.assertEqual(result.mentions, [])

    async def test_factory_rejects_ollama_provider(
        self,
    ):
        settings = build_test_settings(
            provider="ollama"
        )

        with self.assertRaises(ConfigurationError):
            create_llm_provider(
                settings=settings
            )

    async def test_factory_creates_cloud_api_provider(
        self,
    ):
        if CloudAPIProvider is None:
            self.skipTest("openai package is not installed")

        settings = replace(
            build_test_settings(),
            provider="cloud_api",
            cloud_api_base_url="https://api.deepseek.com",
            cloud_api_key="test-key",
            cloud_api_model="deepseek-v4-pro",
        )

        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=AsyncMock()
                )
            ),
            close=AsyncMock(),
        )
        provider = CloudAPIProvider(
            settings=settings,
            client=client,
        )

        self.assertIsInstance(
            provider,
            CloudAPIProvider,
        )
        self.assertEqual(
            provider.provider_name,
            "cloud_api",
        )

        await provider.close()

    async def test_factory_creates_deepseek_provider(
        self,
    ):
        if CloudAPIProvider is None:
            self.skipTest("openai package is not installed")

        settings = replace(
            build_test_settings(),
            provider="deepseek",
            cloud_api_base_url="https://api.deepseek.com",
            cloud_api_key="test-key",
            cloud_api_model="deepseek-v4-pro",
        )

        provider = create_llm_provider(
            settings=settings
        )

        self.assertIsInstance(
            provider,
            CloudAPIProvider,
        )
        self.assertEqual(
            provider.provider_name,
            "deepseek",
        )

        await provider.close()


if __name__ == "__main__":
    unittest.main()
