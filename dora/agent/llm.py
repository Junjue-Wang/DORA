"""LangChain chat models from a DORA model config.

A model config is a JSON object such as::

    {"name": "gemini-3-flash", "provider": "gemini", "model": "gemini-3-flash-preview",
     "generate_args": {"temperature": 0.1, "max_tokens": 4096}}

``provider`` is ``gemini``, ``openai`` (any OpenAI-compatible endpoint, set ``base_url``) or
``azure``. API keys are read from the environment (``api_key_env``, default per provider)
and never stored in configs.
"""
import os
from typing import Any, Dict

DEFAULT_KEY_ENV = {"gemini": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY", "azure": "AZURE_OPENAI_API_KEY"}


def _api_key(cfg: Dict[str, Any], provider: str) -> str:
    env = cfg.get("api_key_env", DEFAULT_KEY_ENV[provider])
    key = os.environ.get(env, "").strip()
    if not key and provider == "azure":
        key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise ValueError(f"Model '{cfg.get('name')}' needs an API key: set the {env} environment variable.")
    return key


def _extra_body(generate_args: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in generate_args.items() if k not in {"temperature", "max_tokens"}}


def build_chat_model(cfg: Dict[str, Any], request_timeout: float = 120):
    provider = cfg.get("provider", "openai")
    args = dict(cfg.get("generate_args") or {})

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = {"model": cfg["model"], "google_api_key": _api_key(cfg, provider),
                  "temperature": args.get("temperature", 0.1)}
        max_tokens = args.get("max_tokens", args.get("max_completion_tokens"))
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens
        kwargs.update({k: v for k, v in args.items() if k not in {"temperature", "max_tokens", "max_completion_tokens"}})
        return ChatGoogleGenerativeAI(**kwargs)

    if provider == "azure":
        from langchain_openai import AzureChatOpenAI

        kwargs = {
            "azure_endpoint": cfg.get("azure_endpoint") or os.environ["AZURE_OPENAI_ENDPOINT"],
            "azure_deployment": cfg.get("azure_deployment") or os.environ.get("AZURE_OPENAI_DEPLOYMENT", cfg["model"]),
            "api_version": cfg.get("api_version") or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            "api_key": _api_key(cfg, provider),
            "temperature": args.get("temperature", 0.1),
            "request_timeout": request_timeout,
        }
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs = {"model": cfg["model"], "api_key": _api_key(cfg, provider),
                  "temperature": args.get("temperature", 0.1), "request_timeout": request_timeout}
        if cfg.get("base_url"):
            kwargs["base_url"] = cfg["base_url"]
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if args.get("max_tokens") is not None:
        kwargs["max_tokens"] = args["max_tokens"]
    if _extra_body(args):
        kwargs["extra_body"] = _extra_body(args)
    return AzureChatOpenAI(**kwargs) if provider == "azure" else ChatOpenAI(**kwargs)
