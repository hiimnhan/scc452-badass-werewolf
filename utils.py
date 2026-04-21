from __future__ import annotations  # MUST be the absolute first line!
from langchain_core.language_models import BaseChatModel
from pathlib import Path
from config import LLM_BASE_CONFIG, MODEL_PROVIDERS
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env


def get_llm(model_name: str, api_key=os.environ["OPENAI_API_KEY"], **kwargs) -> BaseChatModel:
    provider = MODEL_PROVIDERS.get(model_name)
    if not provider:
        raise ValueError(f"Model '{model_name}' is not supported. Choose from: {list(MODEL_PROVIDERS.keys())}")

    return provider(model_name=model_name, **{**LLM_BASE_CONFIG, **kwargs})


def write_to_file(path: Path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, list):
        content = "\n".join(str(item) for item in content)
    elif not isinstance(content, str):
        content = str(content)
    path.write_text(content)
