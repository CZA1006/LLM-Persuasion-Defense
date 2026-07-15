"""Check configured API backends with a minimal request."""

from src.utils import _get_cfg, chat_once

def ping(provider: str) -> None:
    cfg = _get_cfg(override_provider=provider)
    print(f"[{cfg.provider}] model={cfg.model} base_url={cfg.base_url or 'official'}")
    out = chat_once("Reply with 'OK' only.", system="Be terse.", provider=provider)
    print("Response:", out)

if __name__ == "__main__":
    for p in ("openai", "deepseek"):
        try:
            ping(p)
        except Exception as exc:
            print(f"FAILED [{p}]: {exc}")
