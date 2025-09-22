import os, random, numpy as np
from dotenv import load_dotenv

# Optional backends
_BACKEND_OPENAI = "openai"
_BACKEND_AZURE  = "azure_openai"
_BACKEND_HF     = "hf"

def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def _chat_openai(prompt: str, system: str|None, model_name: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=128
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"


def _chat_azure_openai(prompt: str, system: str|None, deployment: str, endpoint: str) -> str:
    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-06-01",
            azure_endpoint=endpoint
        )
        messages = []
        if system:
            messages.append({"role":"system","content":system})
        messages.append({"role":"user","content":prompt})
        resp = client.chat.completions.create(model=deployment, messages=messages, temperature=0.0, max_tokens=128)
        return resp.choices[0].message.content.strip()
    except Exception:
        return "I don't know."

def _chat_hf(prompt: str, system: str|None, model_id: str, max_new_tokens: int = 128) -> str:
    """
    Simple HF text-generation wrapper (no chat template handling here).
    For instruct models, they often work fine with plain text input.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tok = AutoTokenizer.from_pretrained(model_id, use_auth_token=os.getenv("HF_TOKEN") or None)
        mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else None, device_map="auto")
        text = ((system + "\n\n") if system else "") + prompt
        inputs = tok(text, return_tensors="pt").to(mdl.device)
        out = mdl.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0)
        s = tok.decode(out[0], skip_special_tokens=True)
        # return only continuation
        return s[len(text):].strip() or "I don't know."
    except Exception:
        return "I don't know."

def chat_once(prompt: str, system: str|None = None, model: str = "auto") -> str:
    """
    Unified chat wrapper.
    Select backend via .env BACKEND = openai | azure_openai | hf
    """
    load_dotenv(override=True)
    backend = os.getenv("BACKEND", "").strip().lower()

    if backend == _BACKEND_OPENAI:
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return _chat_openai(prompt, system, model_name)

    if backend == _BACKEND_AZURE:
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        return _chat_azure_openai(prompt, system, deployment, endpoint)

    if backend == _BACKEND_HF:
        model_id = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
        max_new  = int(os.getenv("HF_MAX_NEW_TOKENS", "128"))
        return _chat_hf(prompt, system, model_id, max_new)

    # Fallback (no backend configured)
    return "I don't know."
