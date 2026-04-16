from openai import OpenAI
from app.core.config import settings_object as settings

grok_openai_client= OpenAI(base_url=settings.SERVICE_API_URL, api_key=settings.SERVICE_API_KEY)
openai_client= OpenAI(api_key=settings.OPENAI_API_KEY)

def get_grok_client():
    return grok_openai_client

def get_grok_clientaa():
    return grok_openai_client

def get_grok_clientaa_bishwajit():
    return grok_openai_client

def get_grok_clientaa_bishwajit_garai():
    return grok_openai_client