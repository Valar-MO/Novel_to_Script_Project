
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv


# backend/config.py
# parents[0] -> backend
# parents[1] -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE_PATH = PROJECT_ROOT / ".env"


LLMProviderName = Literal[
    "mock",
    "cloud_api",
    "deepseek",
]

SUPPORTED_LLM_PROVIDERS: set[str] = {
    "mock",
    "cloud_api",
    "deepseek",
}


class ConfigurationError(ValueError):
    """应用环境配置不合法。"""


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """Novel2Script 的大模型相关配置。"""

    provider: LLMProviderName

    temperature: float

    cloud_api_base_url: str | None
    cloud_api_key: str | None
    cloud_api_model: str | None
    cloud_api_timeout_seconds: float
    cloud_api_reasoning_effort: str | None
    cloud_api_thinking_enabled: bool

    @property
    def uses_mock(self) -> bool:
        """当前是否使用测试 Mock Provider。"""

        return self.provider == "mock"

    @property
    def uses_cloud_api(self) -> bool:
        """当前是否配置为云端 API。"""

        return self.provider in {
            "cloud_api",
            "deepseek",
        }


def _read_string(
    name: str,
    default: str | None = None,
    *,
    allow_empty: bool = False,
) -> str | None:
    """读取字符串环境变量，并清理首尾空白。"""

    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    normalized_value = raw_value.strip()

    if not normalized_value and not allow_empty:
        return default

    return normalized_value


def _read_float(
    name: str,
    default: float,
) -> float:
    """读取浮点型环境变量。"""

    raw_value = os.getenv(name)

    if raw_value is None or not raw_value.strip():
        return default

    try:
        return float(raw_value)
    except ValueError as error:
        raise ConfigurationError(
            f"环境变量 {name} 必须是数字，"
            f"当前值为：{raw_value!r}"
        ) from error


def _read_bool(
    name: str,
    default: bool,
) -> bool:
    raw_value = os.getenv(name)

    if raw_value is None or not raw_value.strip():
        return default

    normalized_value = raw_value.strip().lower()

    if normalized_value in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True

    if normalized_value in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False

    raise ConfigurationError(
        f"Environment variable {name} must be a boolean value; "
        f"got {raw_value!r}."
    )


def _normalize_base_url(url: str) -> str:
    """移除 URL 末尾多余的斜杠。"""

    normalized_url = url.strip().rstrip("/")

    if not normalized_url:
        raise ConfigurationError(
            "模型服务地址不能为空。"
        )

    if not (
        normalized_url.startswith("http://")
        or normalized_url.startswith("https://")
    ):
        raise ConfigurationError(
            "模型服务地址必须以 http:// "
            "或 https:// 开头。"
        )

    return normalized_url


@lru_cache(maxsize=1)
def get_llm_settings() -> LLMSettings:
    """
    读取并缓存 LLM 配置。

    默认从项目根目录下的 .env 加载配置。
    系统环境变量优先级高于 .env。
    """

    load_dotenv(
        dotenv_path=ENV_FILE_PATH,
        override=False,
    )

    provider_value = (
        _read_string(
            "LLM_PROVIDER",
            "deepseek",
        )
        or "deepseek"
    ).lower()

    if provider_value == "ollama":
        provider_value = "deepseek"

    if provider_value not in SUPPORTED_LLM_PROVIDERS:
        supported_values = ", ".join(
            sorted(SUPPORTED_LLM_PROVIDERS)
        )

        raise ConfigurationError(
            "不支持的 LLM_PROVIDER："
            f"{provider_value!r}。"
            f"可选值：{supported_values}。"
        )

    temperature = _read_float(
        "LLM_TEMPERATURE",
        0.0,
    )

    if not 0 <= temperature <= 2:
        raise ConfigurationError(
            "LLM_TEMPERATURE 必须位于 0 到 2 之间。"
        )

    cloud_api_base_url = (
        _read_string(
            "LLM_API_BASE_URL",
            None,
        )
        or _read_string(
            "DEEPSEEK_BASE_URL",
            None,
        )
    )

    if cloud_api_base_url is None and provider_value == "deepseek":
        cloud_api_base_url = "https://api.deepseek.com"

    if cloud_api_base_url is not None:
        cloud_api_base_url = _normalize_base_url(
            cloud_api_base_url
        )

    cloud_api_key = (
        _read_string(
            "LLM_API_KEY",
            None,
        )
        or _read_string(
            "DEEPSEEK_API_KEY",
            None,
        )
    )

    cloud_api_model = (
        _read_string(
            "LLM_API_MODEL",
            None,
        )
        or _read_string(
            "DEEPSEEK_MODEL",
            None,
        )
    )

    if cloud_api_model is None and provider_value == "deepseek":
        cloud_api_model = "deepseek-v4-pro"

    cloud_api_timeout_seconds = _read_float(
        "LLM_API_TIMEOUT_SECONDS",
        180.0,
    )

    if cloud_api_timeout_seconds <= 0:
        raise ConfigurationError(
            "LLM_API_TIMEOUT_SECONDS 蹇呴』澶т簬 0銆?"
        )

    cloud_api_reasoning_effort = _read_string(
        "LLM_API_REASONING_EFFORT",
        None,
    )

    cloud_api_thinking_enabled = _read_bool(
        "LLM_API_THINKING_ENABLED",
        provider_value == "deepseek",
    )

    return LLMSettings(
        provider=provider_value,
        temperature=temperature,
        cloud_api_base_url=cloud_api_base_url,
        cloud_api_key=cloud_api_key,
        cloud_api_model=cloud_api_model,
        cloud_api_timeout_seconds=(
            cloud_api_timeout_seconds
        ),
        cloud_api_reasoning_effort=(
            cloud_api_reasoning_effort
        ),
        cloud_api_thinking_enabled=(
            cloud_api_thinking_enabled
        ),
    )


def clear_llm_settings_cache() -> None:
    """
    清除配置缓存。

    主要用于单元测试中临时修改环境变量后重新加载配置。
    """

    get_llm_settings.cache_clear()

