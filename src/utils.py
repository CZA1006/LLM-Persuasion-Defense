# src/utils.py
# -*- coding: utf-8 -*-
"""
Utility helpers for API-backed chat calls, seeding, and timing.

Key additions (backward compatible):
- chat_once(..., model_override=None, return_meta=False)
  * model_override: temporarily override the model used for ONE call.
  * return_meta: if True, return (text, meta) with usage & latency info.
- Provider-agnostic OpenAI-compatible client (works for OpenAI, DeepSeek, self-hosted OAI-compatible).

Typical usage (old):
    text = chat_once(system, user, provider="openai", model="gpt-4o")

New usage (optional):
    text, meta = chat_once(system, user, provider="openai", model="gpt-4o",
                           model_override="gpt-4o-mini", return_meta=True)
"""

from __future__ import annotations

import os
import time
import random
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# OpenAI SDK (>=1.0). Most OAI-compatible services (e.g., DeepSeek) work by setting base_url + api_key.
try:
    from openai import OpenAI
except Exception as _e:  # pragma: no cover
    OpenAI = None  # type: ignore

# Optional: numpy / torch seeding if present
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

LOGGER = logging.getLogger("utils")
if not LOGGER.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    LOGGER.addHandler(_h)
LOGGER.setLevel(logging.INFO)


# -------------------------
# Seeding
# -------------------------

def seed_everything(seed: Optional[int] = None) -> None:
    """
    Seed python, numpy, and torch (if available) for best-effort reproducibility.
    """
    if seed is None:
        return
    random.seed(seed)
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass
    if np is not None:
        try:
            np.random.seed(seed)
        except Exception:
            pass
    if torch is not None:
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True  # type: ignore
            torch.backends.cudnn.benchmark = False  # type: ignore
        except Exception:
            pass


# -------------------------
# Environment helpers
# -------------------------

def env_or(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Read environment variable with optional default.
    """
    val = os.getenv(key)
    return val if (val is not None and val != "") else default


# -------------------------
# OpenAI-compatible client
# -------------------------

@dataclass
class ClientConfig:
    provider: str = "openai"        # "openai" | "deepseek" | "<oai-compatible>"
    base_url: Optional[str] = None  # override if using a proxy or a self-hosted endpoint
    api_key: Optional[str] = None   # if None, will read from env depending on provider
    timeout: float = 60.0           # seconds

def _default_api_key(provider: str) -> Optional[str]:
    """
    Resolve API key from environment by provider.
    """
    p = (provider or "openai").lower()
    if p == "openai":
        return env_or("OPENAI_API_KEY")
    if p == "deepseek":
        return env_or("DEEPSEEK_API_KEY")
    # generic fallback
    return env_or("OPENAI_API_KEY") or env_or("API_KEY")


def _default_base_url(provider: str, base_url: Optional[str]) -> Optional[str]:
    """
    Resolve base_url. If user provided, keep it; otherwise provide sensible default for some providers.
    """
    if base_url:
        return base_url
    p = (provider or "openai").lower()
    if p == "deepseek":
        # deepseek official endpoint is OpenAI-compatible:
        return env_or("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    # openai default: None -> SDK uses https://api.openai.com/v1
    return None


def get_oai_client(cfg: ClientConfig) -> OpenAI:
    """
    Return an OpenAI-compatible client. Raises if OpenAI SDK not available.
    """
    if OpenAI is None:
        raise RuntimeError("openai python SDK not installed. Please `pip install openai>=1.0.0`")
    api_key = cfg.api_key or _default_api_key(cfg.provider)
    base_url = _default_base_url(cfg.provider, cfg.base_url)
    if not api_key:
        raise RuntimeError(f"Missing API key for provider={cfg.provider}. "
                           f"Set OPENAI_API_KEY or DEEPSEEK_API_KEY accordingly.")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=cfg.timeout)  # type: ignore
    return client


# -------------------------
# Message builder
# -------------------------

def build_messages(
    system: Optional[str],
    user: Optional[str],
    messages: Optional[List[Dict[str, str]]] = None
) -> List[Dict[str, str]]:
    """
    Build an OpenAI-style messages list.
    If `messages` is provided, it will be used as-is (but we still allow prepending a system message).
    Otherwise, compose from `system` + `user`.
    """
    if messages is not None:
        if system:
            return [{"role": "system", "content": system}] + messages
        return list(messages)
    msgs: List[Dict[str, str]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    if user:
        msgs.append({"role": "user", "content": user})
    return msgs


# -------------------------
# Chat call with retries
# -------------------------

def _now_ms() -> int:
    import time as _t
    return int(_t.time() * 1000)


def _sleep_backoff(attempt: int, base: float, jitter: float = 0.25) -> None:
    """
    Exponential backoff with jitter.
    attempt: 0-based
    """
    import time as _t
    delay = base * (2 ** attempt)
    # jitter in [1 - jitter, 1 + jitter]
    factor = 1.0 + random.uniform(-jitter, jitter)
    _t.sleep(max(0.0, delay * factor))


def _extract_text_from_response(resp: Any) -> str:
    """
    Support both Responses API and Chat Completions if needed.
    Prefer chat.completions since most OAI-compatible vendors expose it.
    """
    try:
        # OpenAI chat.completions
        return resp.choices[0].message.content or ""
    except Exception:
        # Try "responses" api shape (rare in OAI-compatible vendors)
        try:
            return resp.output_text  # type: ignore
        except Exception:
            return ""


def chat_once(
    system: Optional[str],
    user: Optional[str],
    *,
    provider: Optional[str] = "openai",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    timeout: float = 60.0,
    retries: int = 3,
    backoff_base: float = 0.8,
    extra_headers: Optional[Dict[str, str]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    # New:
    model_override: Optional[str] = None,
    return_meta: bool = False,
) -> Union[str, Tuple[str, Dict[str, Any]]]:
    """
    Make a single chat request with retry/backoff.
    If return_meta=True, returns (text, meta).

    Meta dict includes:
      - usage (prompt_tokens, completion_tokens, total_tokens) if available
      - latency_ms
      - model (effective)
      - provider, base_url
      - retry_count
    """
    if seed is not None:
        # Not all providers honor "seed", but we set it for those who do.
        try:
            seed_everything(seed)
        except Exception:
            pass

    eff_model = model_override or model
    if not eff_model:
        raise ValueError("chat_once: `model` is required (or pass `model_override`).")

    cfg = ClientConfig(
        provider=(provider or "openai"),
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )
    client = get_oai_client(cfg)

    msgs = build_messages(system=system, user=user, messages=messages)

    last_err: Optional[Exception] = None
    t0 = _now_ms()
    attempt = 0
    resp = None

    while attempt <= retries:
        try:
            # OpenAI Chat Completions API
            resp = client.chat.completions.create(
                model=eff_model,
                messages=msgs,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                tools=tools,
                tool_choice=tool_choice,
                extra_headers=extra_headers,  # type: ignore
            )
            break
        except Exception as e:
            last_err = e
            if attempt >= retries:
                LOGGER.error("chat_once failed after %d retries: %s", retries, repr(e))
                raise
            _sleep_backoff(attempt, backoff_base)
            attempt += 1

    t1 = _now_ms()
    latency_ms = t1 - t0
    text = _extract_text_from_response(resp)

    # Usage extraction (best effort; schema varies a bit across vendors)
    usage = None
    try:
        if hasattr(resp, "usage") and resp.usage is not None:
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None),
            }
    except Exception:
        usage = None

    if not return_meta:
        return text

    meta: Dict[str, Any] = {
        "usage": usage,
        "latency_ms": latency_ms,
        "model": eff_model,
        "provider": cfg.provider,
        "base_url": _default_base_url(cfg.provider, cfg.base_url),
        "retry_count": attempt,
    }
    return text, meta


# -------------------------
# Convenience wrappers
# -------------------------

def simple_chat(system: str, user: str, **kwargs) -> str:
    """
    Small convenience to keep older call sites readable.
    """
    return chat_once(system, user, **kwargs)  # type: ignore[return-value]


def chat_with_meta(system: str, user: str, **kwargs) -> Tuple[str, Dict[str, Any]]:
    """
    Always return (text, meta).
    """
    kwargs["return_meta"] = True
    return chat_once(system, user, **kwargs)  # type: ignore[return-value]
