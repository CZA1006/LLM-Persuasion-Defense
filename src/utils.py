# src/utils.py
from __future__ import annotations
import os, time, random, numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

from dotenv import load_dotenv, find_dotenv

# OpenAI SDK 支持自定义 base_url；可用于代理与 DeepSeek
from openai import OpenAI
from openai import APIConnectionError, RateLimitError, APIStatusError

# -----------------------------------------------------------------------------
# 自动加载 .env（向上查找项目根目录的 .env），避免每次手动 `source .env`
# -----------------------------------------------------------------------------
try:
    _dotenv_path = find_dotenv(usecwd=True)
    if _dotenv_path:
        load_dotenv(_dotenv_path, override=False)
    else:
        load_dotenv(override=False)
except Exception:
    # 加载失败不致命，后续仍可依赖已有环境变量
    pass

# -----------------------------------------------------------------------------
# 随机种子
# -----------------------------------------------------------------------------
def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

# -----------------------------------------------------------------------------
# 配置结构
# -----------------------------------------------------------------------------
@dataclass
class BackendConfig:
    provider: str
    api_key: str
    base_url: Optional[str]
    model: str
    temperature: float
    max_tokens: int
    timeout: int

# -----------------------------------------------------------------------------
# 环境变量辅助
# -----------------------------------------------------------------------------
def _env(k: str, default: str = "") -> str:
    v = os.getenv(k, default)
    return (v or "").strip()

def _env_opt(k: str) -> Optional[str]:
    v = _env(k, "")
    return v or None

def _mask(key: str) -> str:
    if not key:
        return ""
    return f"{'*' * max(0, len(key)-4)}{key[-4:]}"

def _require_key(provider: str, key: str) -> None:
    if key:
        return
    if provider == "openai":
        raise RuntimeError(
            "Missing API key for provider=openai. "
            "Set OPENAI_API_KEY in your environment or in the .env file."
        )
    if provider == "deepseek":
        raise RuntimeError(
            "Missing API key for provider=deepseek. "
            "Set DEEPSEEK_API_KEY in your environment or in the .env file."
        )
    raise RuntimeError("Missing API key.")

# -----------------------------------------------------------------------------
# 读取后端配置（支持 CLI 覆盖：见 run_ablation.apply_backend_overrides）
# -----------------------------------------------------------------------------
def _get_cfg(override_provider: Optional[str] = None,
             override_model: Optional[str] = None) -> BackendConfig:
    provider = (override_provider or _env("PROVIDER", "openai")).lower()
    if provider not in {"openai", "deepseek"}:
        raise ValueError(f"Unsupported PROVIDER: {provider}")

    if provider == "openai":
        cfg = BackendConfig(
            provider="openai",
            api_key=_env("OPENAI_API_KEY"),
            base_url=_env_opt("OPENAI_BASE_URL"),  # e.g., https://api.laozhang.ai/v1
            model=override_model or _env("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=float(_env("GEN_TEMPERATURE", "0.0")),
            max_tokens=int(_env("GEN_MAX_TOKENS", "512")),
            timeout=int(_env("REQUEST_TIMEOUT", "60")),
        )
    else:
        cfg = BackendConfig(
            provider="deepseek",
            api_key=_env("DEEPSEEK_API_KEY"),
            base_url=_env_opt("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1",
            model=override_model or _env("DEEPSEEK_MODEL", "deepseek-chat"),
            temperature=float(_env("GEN_TEMPERATURE", "0.0")),
            max_tokens=int(_env("GEN_MAX_TOKENS", "512")),
            timeout=int(_env("REQUEST_TIMEOUT", "60")),
        )

    # 关键：在这里统一校验 API Key，报错更早更清晰
    _require_key(cfg.provider, cfg.api_key)
    return cfg

# -----------------------------------------------------------------------------
# 客户端构造（兼容旧函数名 get_oai_client）
# -----------------------------------------------------------------------------
def _make_client(cfg: BackendConfig) -> OpenAI:
    # OpenAI SDK v1：OpenAI(api_key=..., base_url=...)
    return OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

# 向后兼容：一些旧代码可能直接调用 get_oai_client(cfg)
def get_oai_client(cfg: Optional[BackendConfig] = None) -> OpenAI:
    """
    Back-compat shim. If cfg is None, build one from env (provider=openai).
    """
    if cfg is None:
        cfg = _get_cfg(override_provider="openai")
    _require_key(cfg.provider, cfg.api_key)
    return _make_client(cfg)

# -----------------------------------------------------------------------------
# 核心聊天调用
# -----------------------------------------------------------------------------
_MAX_RETRIES = 5

def chat_once(
    prompt: str,
    system: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    provider: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Single, non-streaming chat call. Defense text should go ONLY in `system`.

    Args:
        prompt: user content
        system: system content (用于 defense/override)
        model: 覆盖使用的模型名（None 使用 env/默认）
        provider: 覆盖后端（'openai' 或 'deepseek'）
        extra: 透传到 client.chat.completions.create 的其它参数
    """
    if model is not None and str(model).lower() == "auto":
        model = None

    cfg = _get_cfg(override_provider=provider, override_model=model)
    client = _make_client(cfg)

    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    t = cfg.temperature if temperature is None else float(temperature)
    mt = cfg.max_tokens if max_tokens is None else int(max_tokens)

    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=messages,
                temperature=t,
                max_tokens=mt,
                **(extra or {}),
            )
            return resp.choices[0].message.content or ""
        except (RateLimitError, APIConnectionError, APIStatusError) as e:
            if attempt == _MAX_RETRIES - 1:
                # 向上抛出，让外层清晰看到是哪类错误
                raise
            # 指数 + 抖动退避
            time.sleep(1.0 + 0.8 * attempt + random.random())
    return ""

def chat_once_pair(
    *,
    system: Optional[str] = None,
    user: str = "",
    model_override: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    provider: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    兼容 X-Team 风格的 (text, meta) 返回；旧代码不依赖它，纯增量。
    meta 至少包含 {"provider": ..., "model": ...}
    """
    text = chat_once(
        prompt=user, system=system, model=model_override,
        temperature=temperature, max_tokens=max_tokens,
        provider=provider, extra=extra,
    ) or ""
    meta = {
        "provider": (provider or (_env("PROVIDER", "openai"))).lower(),
        "model": model_override or _env("OPENAI_MODEL" if (provider or _env("PROVIDER","openai")).lower()=="openai" else "DEEPSEEK_MODEL", ""),
    }
    return text, meta

# -----------------------------------------------------------------------------
# 便捷打印当前后端（可选：调试时使用）
# -----------------------------------------------------------------------------
def debug_backend_banner() -> str:
    prov = _env("PROVIDER", "openai").lower()
    if prov == "openai":
        key = _env("OPENAI_API_KEY")
        base = _env_opt("OPENAI_BASE_URL") or "default"
        model = _env("OPENAI_MODEL", "gpt-4o-mini")
    else:
        key = _env("DEEPSEEK_API_KEY")
        base = _env_opt("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1"
        model = _env("DEEPSEEK_MODEL", "deepseek-chat")
    return f"[Backend] provider={prov} model={model} base_url={base} key={_mask(key)}"
