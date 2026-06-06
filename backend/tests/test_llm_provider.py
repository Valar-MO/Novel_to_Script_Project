
import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

from ollama import ResponseError

from backend.config import (
    ConfigurationError,
    LLMSettings,
)
from backend.llm.base import (
    LLMModelNotFoundError,
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
from backend.llm.ollama_provider import (
    OllamaProvider,
)
from backend.llm.schemas import (
    MentionExtractionOutput,
)


def build_test_settings(
    *,
    provider: str = "ollama",
) -> LLMSettings:
    """创建不依赖本地 .env 的测试配置。"""

    return LLMSettings(
        provider=provider,  # type: ignore[arg-type]
        ollama_base_url=(
            "http://127.0.0.1:11434"
        ),
        ollama_model="qwen3:8b",
        ollama_timeout_seconds=30.0,
        temperature=0.0,
        keep_alive="10m",
        cloud_api_base_url=None,
        cloud_api_key=None,
        cloud_api_model=None,
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


class TestOllamaProvider(
    unittest.IsolatedAsyncioTestCase
):
    """
    使用假的 AsyncClient 测试 OllamaProvider。

    这些测试不会连接真实 Ollama，也不会加载 Qwen3。
    """

    def setUp(self):
        self.settings = build_test_settings()

        self.client = SimpleNamespace(
            chat=AsyncMock(),
            list=AsyncMock(),
            close=AsyncMock(),
        )

        self.provider = OllamaProvider(
            settings=self.settings,
            client=self.client,
        )

    async def test_generate_structured_returns_valid_result(
        self,
    ):
        self.client.chat.return_value = (
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        build_mention_payload(),
                        ensure_ascii=False,
                    )
                )
            )
        )

        result = await (
            self.provider.generate_structured(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是人物提取器。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "韩立又叫二愣子。"
                        ),
                    },
                ],
                response_model=(
                    MentionExtractionOutput
                ),
            )
        )

        self.assertEqual(result.mentions[0].mention_text, "韩立")

        self.client.chat.assert_awaited_once()

        request_kwargs = (
            self.client.chat.await_args.kwargs
        )

        self.assertEqual(
            request_kwargs["model"],
            "qwen3:8b",
        )
        self.assertFalse(
            request_kwargs["stream"]
        )
        self.assertFalse(
            request_kwargs["think"]
        )
        self.assertEqual(
            request_kwargs["keep_alive"],
            "10m",
        )
        self.assertEqual(
            request_kwargs["options"],
            {
                "temperature": 0.0,
            },
        )
        self.assertIn(
            "properties",
            request_kwargs["format"],
        )

    async def test_valid_mention_result_does_not_trigger_repair(
        self,
    ):
        self.client.chat.return_value = (
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        build_mention_payload(),
                        ensure_ascii=False,
                    )
                )
            )
        )

        result = await self.provider.generate_structured(
            messages=[
                {
                    "role": "user",
                    "content": "韩立走进山谷。",
                }
            ],
            response_model=MentionExtractionOutput,
            temperature=0,
        )

        self.assertEqual(result.mentions[0].mention_text, "韩立")
        self.assertEqual(
            self.client.chat.await_count,
            1,
        )

    async def test_fenced_json_result_is_cleaned_before_validation(
        self,
    ):
        self.client.chat.return_value = (
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        "```json\n"
                        f"{json.dumps(build_mention_payload(), ensure_ascii=False)}"
                        "\n```"
                    )
                )
            )
        )

        result = await self.provider.generate_structured(
            messages=[
                {
                    "role": "user",
                    "content": "韩立走进山谷。",
                }
            ],
            response_model=MentionExtractionOutput,
            temperature=0,
        )

        self.assertEqual(result.mentions[0].mention_text, "韩立")
        self.assertEqual(
            self.client.chat.await_count,
            1,
        )

    async def test_trailing_text_after_json_is_cleaned_before_validation(
        self,
    ):
        self.client.chat.return_value = (
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        json.dumps(
                            build_mention_payload(),
                            ensure_ascii=False,
                        )
                        + "\n\n下面是对这些数据的分析。"
                    )
                )
            )
        )

        result = await self.provider.generate_structured(
            messages=[
                {
                    "role": "user",
                    "content": "韩立走进山谷。",
                }
            ],
            response_model=MentionExtractionOutput,
            temperature=0,
        )

        self.assertEqual(result.mentions[0].mention_text, "韩立")
        self.assertEqual(
            self.client.chat.await_count,
            1,
        )

    async def test_invalid_mention_result_triggers_repair_success(
        self,
    ):
        self.client.chat.side_effect = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        {
                            "mentions": [
                                {
                                    "mention_type": "character",
                                    "name": "韩立",
                                }
                            ]
                        },
                        ensure_ascii=False,
                    )
                )
            ),
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        build_mention_payload(),
                        ensure_ascii=False,
                    )
                )
            ),
        ]

        result = await self.provider.generate_structured(
            messages=[
                {
                    "role": "user",
                    "content": "韩立走进山谷。",
                }
            ],
            response_model=MentionExtractionOutput,
            temperature=0,
        )

        self.assertEqual(
            self.client.chat.await_count,
            2,
        )
        self.assertEqual(result.mentions[0].mention_text, "韩立")

        repair_kwargs = (
            self.client.chat.await_args_list[1].kwargs
        )
        self.assertEqual(
            repair_kwargs["options"],
            {
                "temperature": 0,
            },
        )
        self.assertIn(
            "原始输出",
            repair_kwargs["messages"][-1]["content"],
        )
        self.assertEqual(
            repair_kwargs["messages"][0]["role"],
            "system",
        )
        self.assertEqual(
            repair_kwargs["messages"][1]["role"],
            "user",
        )
        self.assertEqual(
            len(repair_kwargs["messages"]),
            2,
        )
        self.assertIn("原始输出", repair_kwargs["messages"][-1]["content"])

    async def test_invalid_mention_result_and_repair_raise_error(
        self,
    ):
        invalid_payload = {
            "mentions": [
                {
                    "mention_type": "character",
                    "name": "韩立",
                }
            ]
        }

        self.client.chat.side_effect = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        invalid_payload,
                        ensure_ascii=False,
                    )
                )
            ),
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        invalid_payload,
                        ensure_ascii=False,
                    )
                )
            ),
        ]

        with self.assertRaises(
            LLMResponseValidationError
        ):
            await self.provider.generate_structured(
                messages=[
                    {
                        "role": "user",
                        "content": "韩立走进山谷。",
                    }
                ],
                response_model=MentionExtractionOutput,
                temperature=0,
            )

        self.assertEqual(
            self.client.chat.await_count,
            2,
        )

    async def test_temperature_override_is_used(
        self,
    ):
        self.client.chat.return_value = (
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        {
                            "mentions": [],
                        }
                    )
                )
            )
        )

        await self.provider.generate_structured(
            messages=[
                {
                    "role": "user",
                    "content": "测试。",
                }
            ],
            response_model=(
                MentionExtractionOutput
            ),
            temperature=0.2,
        )

        request_kwargs = (
            self.client.chat.await_args.kwargs
        )

        self.assertEqual(
            request_kwargs["options"],
            {
                "temperature": 0.2,
            },
        )

    async def test_model_not_found_is_converted(
        self,
    ):
        self.client.chat.side_effect = (
            ResponseError(
                "model not found",
                404,
            )
        )

        with self.assertRaises(
            LLMModelNotFoundError
        ):
            await self.provider.generate_structured(
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

    async def test_connection_failure_is_converted(
        self,
    ):
        self.client.chat.side_effect = (
            ConnectionError(
                "connection refused"
            )
        )

        with self.assertRaises(
            LLMProviderUnavailableError
        ):
            await self.provider.generate_structured(
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

    async def test_invalid_json_is_converted(
        self,
    ):
        self.client.chat.return_value = (
            SimpleNamespace(
                message=SimpleNamespace(
                    content="不是合法 JSON"
                )
            )
        )

        with self.assertRaises(
            LLMResponseValidationError
        ):
            await self.provider.generate_structured(
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

    async def test_health_check_reports_installed_model(
        self,
    ):
        self.client.list.return_value = (
            SimpleNamespace(
                models=[
                    SimpleNamespace(
                        model="qwen3:8b"
                    )
                ]
            )
        )

        health = (
            await self.provider.health_check()
        )

        self.assertTrue(
            health.available
        )
        self.assertEqual(
            health.provider,
            "ollama",
        )
        self.assertEqual(
            health.model,
            "qwen3:8b",
        )

    async def test_health_check_reports_missing_model(
        self,
    ):
        self.client.list.return_value = (
            SimpleNamespace(
                models=[
                    SimpleNamespace(
                        model="qwen3:4b"
                    )
                ]
            )
        )

        health = (
            await self.provider.health_check()
        )

        self.assertFalse(
            health.available
        )
        self.assertIn(
            "没有找到模型",
            health.detail,
        )

    async def test_closed_provider_rejects_requests(
        self,
    ):
        # 注入的测试客户端不归 Provider 所有，
        # 但 Provider 本身仍会标记为已关闭。
        await self.provider.close()

        with self.assertRaises(
            LLMProviderUnavailableError
        ):
            await self.provider.generate_structured(
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

    async def test_factory_creates_ollama_provider(
        self,
    ):
        settings = build_test_settings(
            provider="ollama"
        )

        provider = create_llm_provider(
            settings=settings
        )

        self.assertIsInstance(
            provider,
            OllamaProvider,
        )

        # 只关闭客户端，不进行真实网络请求。
        await provider.close()

    async def test_cloud_api_is_reserved_but_not_implemented(
        self,
    ):
        settings = replace(
            build_test_settings(),
            provider="cloud_api",
        )

        with self.assertRaises(
            ConfigurationError
        ):
            create_llm_provider(
                settings=settings
            )


if __name__ == "__main__":
    unittest.main()
