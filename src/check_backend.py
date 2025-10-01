# check_backend.py
try:
    from src.utils import chat_once, _get_cfg   # if you use a src/ package
except ImportError:
    from utils import chat_once, _get_cfg       # top-level fallback

def ping(provider: str):
    cfg = _get_cfg(override_provider=provider)
    print(f"[{cfg.provider}] model={cfg.model} base_url={cfg.base_url or 'official'}")
    out = chat_once("Reply with 'OK' only.", system="Be terse.", provider=provider)
    print("Response:", out)

if __name__ == "__main__":
    for p in ("openai", "deepseek"):
        try:
            ping(p)
        except Exception as e:
            print(f"FAILED [{p}]: {e}")
