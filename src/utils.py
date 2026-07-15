"""Shared model-backend and reproducibility utilities."""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import find_dotenv, load_dotenv

from openai import APIConnectionError, RateLimitError, APIStatusError
from openai import OpenAI

# Load the nearest .env file without replacing existing environment variables.
try:
    _dotenv_path = find_dotenv(usecwd=True)
    if _dotenv_path:
        load_dotenv(_dotenv_path, override=False)
    else:
        load_dotenv(override=False)
except Exception:
    # Environment variables remain usable if dotenv discovery fails.
    pass

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

def _get_cfg(
    override_provider: Optional[str] = None,
    override_model: Optional[str] = None,
) -> BackendConfig:
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

    # Validate credentials before constructing a request.
    _require_key(cfg.provider, cfg.api_key)
    return cfg

def _get_attack_cfg(
    override_provider: Optional[str] = None,
    override_model: Optional[str] = None,
) -> BackendConfig:
    """
    Backend config for the Attack LLM.

    Provider precedence:
      1) override_provider
      2) ATTACK_PROVIDER
      3) PROVIDER

    Model and base URL precedence:
      - OpenAI:
          model: override_model -> ATTACK_OPENAI_MODEL -> OPENAI_MODEL
          base: ATTACK_OPENAI_BASE_URL -> OPENAI_BASE_URL
      - DeepSeek:
          model: override_model -> ATTACK_DEEPSEEK_MODEL -> DEEPSEEK_MODEL
          base: ATTACK_DEEPSEEK_BASE_URL -> DEEPSEEK_BASE_URL -> official default
    API keys are inherited from OPENAI_API_KEY or DEEPSEEK_API_KEY.
    """
    base_provider = _env("PROVIDER", "openai")
    provider = (override_provider or _env("ATTACK_PROVIDER", base_provider)).lower()
    if provider not in {"openai", "deepseek"}:
        raise ValueError(f"Unsupported ATTACK_PROVIDER: {provider}")

    if provider == "openai":
        cfg = BackendConfig(
            provider="openai",
            api_key=_env("OPENAI_API_KEY"),
            base_url=(
                _env_opt("ATTACK_OPENAI_BASE_URL")
                or _env_opt("OPENAI_BASE_URL")
            ),
            model=(
                override_model
                or _env("ATTACK_OPENAI_MODEL", "")
                or _env("OPENAI_MODEL", "gpt-4o-mini")
            ),
            temperature=float(_env("GEN_TEMPERATURE", "0.0")),
            max_tokens=int(_env("GEN_MAX_TOKENS", "512")),
            timeout=int(_env("REQUEST_TIMEOUT", "60")),
        )
    else:
        cfg = BackendConfig(
            provider="deepseek",
            api_key=_env("DEEPSEEK_API_KEY"),
            base_url=(
                _env_opt("ATTACK_DEEPSEEK_BASE_URL")
                or _env_opt("DEEPSEEK_BASE_URL")
                or "https://api.deepseek.com/v1"
            ),
            model=(
                override_model
                or _env("ATTACK_DEEPSEEK_MODEL", "")
                or _env("DEEPSEEK_MODEL", "deepseek-chat")
            ),
            temperature=float(_env("GEN_TEMPERATURE", "0.0")),
            max_tokens=int(_env("GEN_MAX_TOKENS", "512")),
            timeout=int(_env("REQUEST_TIMEOUT", "60")),
        )

    _require_key(cfg.provider, cfg.api_key)
    return cfg

def _make_client(cfg: BackendConfig) -> OpenAI:
    return OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

def get_oai_client(cfg: Optional[BackendConfig] = None) -> OpenAI:
    """Compatibility shim that builds an OpenAI configuration by default."""
    if cfg is None:
        cfg = _get_cfg(override_provider="openai")
    _require_key(cfg.provider, cfg.api_key)
    return _make_client(cfg)

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
    Single, non-streaming chat call for the *target* LLM.
    Defense text should go ONLY in `system`.

    Args:
        prompt: user content
        system: system content, including optional defense instructions
        model: model override; ``None`` uses the configured default
        provider: backend override (``openai`` or ``deepseek``)
        extra: additional arguments for ``client.chat.completions.create``
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
        except (RateLimitError, APIConnectionError, APIStatusError):
            if attempt == _MAX_RETRIES - 1:
                raise
            # Linear backoff with jitter.
            time.sleep(1.0 + 0.8 * attempt + random.random())
    return ""

def chat_once_attack(
    prompt: str,
    system: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    provider: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Single, non-streaming chat call for the *Attack LLM*.

    ATTACK_* variables take precedence and otherwise inherit the target backend.
    """
    if model is not None and str(model).lower() == "auto":
        model = None

    cfg = _get_attack_cfg(override_provider=provider, override_model=model)
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
        except (RateLimitError, APIConnectionError, APIStatusError):
            if attempt == _MAX_RETRIES - 1:
                raise
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
    """Return an X-Team-compatible ``(text, metadata)`` pair."""
    text = chat_once(
        prompt=user, system=system, model=model_override,
        temperature=temperature, max_tokens=max_tokens,
        provider=provider, extra=extra,
    ) or ""
    meta = {
        "provider": (provider or (_env("PROVIDER", "openai"))).lower(),
        "model": model_override or _env(
            "OPENAI_MODEL"
            if (provider or _env("PROVIDER", "openai")).lower() == "openai"
            else "DEEPSEEK_MODEL",
            "",
        ),
    }
    return text, meta

def debug_backend_banner() -> str:
    # target
    prov = _env("PROVIDER", "openai").lower()
    if prov == "openai":
        key = _env("OPENAI_API_KEY")
        base = _env_opt("OPENAI_BASE_URL") or "default"
        model = _env("OPENAI_MODEL", "gpt-4o-mini")
    else:
        key = _env("DEEPSEEK_API_KEY")
        base = _env_opt("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1"
        model = _env("DEEPSEEK_MODEL", "deepseek-chat")

    # attack（如果没设就继承 target）
    atk_prov = _env("ATTACK_PROVIDER", prov).lower()
    if atk_prov == "openai":
        atk_model = _env("ATTACK_OPENAI_MODEL", "") or _env("OPENAI_MODEL", "gpt-4o-mini")
        atk_base = _env_opt("ATTACK_OPENAI_BASE_URL") or _env_opt("OPENAI_BASE_URL") or "default"
    else:
        atk_model = _env("ATTACK_DEEPSEEK_MODEL", "") or _env("DEEPSEEK_MODEL", "deepseek-chat")
        atk_base = (
            _env_opt("ATTACK_DEEPSEEK_BASE_URL")
            or _env_opt("DEEPSEEK_BASE_URL")
            or "https://api.deepseek.com/v1"
        )

    return (
        f"[Target backend] provider={prov} model={model} base_url={base} key={_mask(key)}\n"
        f"[Attack backend] provider={atk_prov} model={atk_model or '(inherit)'} "
        f"base_url={atk_base}"
    )
