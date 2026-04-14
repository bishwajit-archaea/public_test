
from app.services.llm.llm_clients import openai_client,grok_openai_client
from app.core.config import settings_object as settings

# All available LLM models grouped by provider type
LLM_MODELS = {
    "grok": [
        {
            "model_name": "meta-llama/llama-4-scout-17b-16e-instruct",
            "display_name": "LLama 4 Scout 17B",
            "model_provider": "grok",
            "client": grok_openai_client,
        },
        {
            "model_name": "moonshotai/kimi-k2-instruct",
            "display_name": "Kimi K2 Instruct",
            "model_provider": "grok",
            "client": grok_openai_client,
        },
        {
            "model_name": "openai/gpt-oss-120b",
            "display_name": "GPT OSS 120B",
            "model_provider": "grok",
            "client": grok_openai_client,
        },
        {
            "model_name": "openai/gpt-oss-safeguard-20b",
            "display_name": "GPT OSS Safeguard 20B",
            "model_provider": "grok",
            "client": grok_openai_client,
        },
        {
            "model_name": "moonshotai/kimi-k2-instruct-0905",
            "display_name": "Kimi K2 Instruct 0905",
            "model_provider": "grok",
            "client": grok_openai_client,
        },
        {
            "model_name": "qwen/qwen3-32b",
            "display_name": "Qwen Qwen3 32B",
            "model_provider": "grok",
            "client": grok_openai_client,
        },
        {
            "model_name": "llama-3.3-70b-versatile",
            "display_name": "LLama 3.3 70B Versatile",
            "model_provider": "grok",
            "client": grok_openai_client,
        },
        {
            "model_name": "llama-4-maverick-17b-128e-instruct",
            "display_name": "LLama 4 Maverick 17B",
            "model_provider": "grok",
            "client": grok_openai_client,
        },
    ],

    "openai": [
        {
            "model_name": "gpt-4o-mini",
            "display_name": "GPT 4O Mini",
            "model_provider": "openai",
            "client": openai_client,
        }
    ]
}