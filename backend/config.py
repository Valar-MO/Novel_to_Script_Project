
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
    "ollama",
    "mock",
    "cloud_api",
]

SUPPORTED_LLM_PROVIDERS: set[str] = {
    "ollama",
    "mock",
    "cloud_api",
}


class ConfigurationError(ValueError):
    """应用环境配置不合法。"""


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """Novel2Script 的大模型相关配置。"""

    provider: LLMProviderName

    ollama_base_url: str
    ollama_model: str
    ollama_timeout_seconds: float

    temperature: float
    keep_alive: str

    cloud_api_base_url: str | None
    cloud_api_key: str | None
    cloud_api_model: str | None

    @property
    def uses_ollama(self) -> bool:
        """当前是否使用本地 Ollama。"""

        return self.provider == "ollama"

    @property
    def uses_mock(self) -> bool:
        """当前是否使用测试 Mock Provider。"""

        return self.provider == "mock"

    @property
    def uses_cloud_api(self) -> bool:
        """当前是否配置为云端 API。"""

        return self.provider == "cloud_api"


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
            "ollama",
        )
        or "ollama"
    ).lower()

    if provider_value not in SUPPORTED_LLM_PROVIDERS:
        supported_values = ", ".join(
            sorted(SUPPORTED_LLM_PROVIDERS)
        )

        raise ConfigurationError(
            "不支持的 LLM_PROVIDER："
            f"{provider_value!r}。"
            f"可选值：{supported_values}。"
        )

    ollama_base_url = _normalize_base_url(
        _read_string(
            "OLLAMA_BASE_URL",
            "http://127.0.0.1:11434",
        )
        or "http://127.0.0.1:11434"
    )

    ollama_model = (
        _read_string(
            "OLLAMA_MODEL",
            "qwen3:8b",
        )
        or "qwen3:8b"
    )

    ollama_timeout_seconds = _read_float(
        "OLLAMA_TIMEOUT_SECONDS",
        180.0,
    )

    if ollama_timeout_seconds <= 0:
        raise ConfigurationError(
            "OLLAMA_TIMEOUT_SECONDS 必须大于 0。"
        )

    temperature = _read_float(
        "LLM_TEMPERATURE",
        0.0,
    )

    if not 0 <= temperature <= 2:
        raise ConfigurationError(
            "LLM_TEMPERATURE 必须位于 0 到 2 之间。"
        )

    keep_alive = (
        _read_string(
            "LLM_KEEP_ALIVE",
            "10m",
        )
        or "10m"
    )

    cloud_api_base_url = _read_string(
        "LLM_API_BASE_URL",
        None,
    )

    if cloud_api_base_url is not None:
        cloud_api_base_url = _normalize_base_url(
            cloud_api_base_url
        )

    cloud_api_key = _read_string(
        "LLM_API_KEY",
        None,
    )

    cloud_api_model = _read_string(
        "LLM_API_MODEL",
        None,
    )

    return LLMSettings(
        provider=provider_value,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
        ollama_timeout_seconds=(
            ollama_timeout_seconds
        ),
        temperature=temperature,
        keep_alive=keep_alive,
        cloud_api_base_url=cloud_api_base_url,
        cloud_api_key=cloud_api_key,
        cloud_api_model=cloud_api_model,
    )


def clear_llm_settings_cache() -> None:
    """
    清除配置缓存。

    主要用于单元测试中临时修改环境变量后重新加载配置。
    """

    get_llm_settings.cache_clear()

