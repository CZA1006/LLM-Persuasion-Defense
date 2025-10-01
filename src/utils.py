# utils.py
from __future__ import annotations
import os, time, random, numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# OpenAI SDK supports custom base_url; works for proxies and DeepSeek
from openai import OpenAI
from openai import APIConnectionError, RateLimitError, APIStatusError

load_dotenv()

def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

@dataclass
class BackendConfig:
    provider: str
    api_key: str
    base_url: Optional[str]
    model: str
    temperature: float
    max_tokens: int
    timeout: int

def _env(k: str, default: str = "") -> str:
    v = os.getenv(k, default)
    return v.strip()

def _env_opt(k: str) -> Optional[str]:
    v = _env(k, "")
    return v or None

def _get_cfg(override_provider: Optional[str] = None,
             override_model: Optional[str] = None) -> BackendConfig:
    provider = (override_provider or _env("PROVIDER", "openai")).lower()
    if provider not in {"openai", "deepseek"}:
        raise ValueError(f"Unsupported PROVIDER: {provider}")

    if provider == "openai":
        return BackendConfig(
            provider="openai",
            api_key=_env("OPENAI_API_KEY"),
            base_url=_env_opt("OPENAI_BASE_URL"),  # e.g., https://api.laozhang.ai/v1
            model=override_model or _env("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=float(_env("GEN_TEMPERATURE", "0.0")),
            max_tokens=int(_env("GEN_MAX_TOKENS", "512")),
            timeout=int(_env("REQUEST_TIMEOUT", "60")),
        )
    else:
        return BackendConfig(
            provider="deepseek",
            api_key=_env("DEEPSEEK_API_KEY"),
            base_url=_env_opt("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1",
            model=override_model or _env("DEEPSEEK_MODEL", "deepseek-chat"),
            temperature=float(_env("GEN_TEMPERATURE", "0.0")),
            max_tokens=int(_env("GEN_MAX_TOKENS", "512")),
            timeout=int(_env("REQUEST_TIMEOUT", "60")),
        )

def _make_client(cfg: BackendConfig) -> OpenAI:
    return OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

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
    """
    cfg = _get_cfg(override_provider=provider, override_model=model)
    client = _make_client(cfg)

    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    t = cfg.temperature if temperature is None else temperature
    mt = cfg.max_tokens if max_tokens is None else max_tokens

    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=messages,
                temperature=t,
                max_tokens=mt,
                **(extra or {}),
            )
            return resp.choices[0].message.content or ""
        except (RateLimitError, APIConnectionError, APIStatusError):
            if attempt == 4:
                raise
            # jittered backoff
            time.sleep(1.0 + 0.8 * attempt + random.random())
    return ""
