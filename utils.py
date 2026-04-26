from __future__ import annotations  # MUST be the absolute first line!
from langchain_core.language_models import BaseChatModel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pathlib import Path
from langchain_openai import ChatOpenAI
from config import LLM_BASE_CONFIG, MODEL_PROVIDERS
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()  # Load environment variables from .env


def get_llm(model_name: str, api_key=os.environ.get("DEEPINFRA_API_KEY"), **kwargs) -> BaseChatModel:
    config = {**LLM_BASE_CONFIG, **kwargs}
    provider = MODEL_PROVIDERS.get(model_name)
    
    if not provider:
        raise ValueError(f"Model '{model_name}' is not supported. Choose from: {list(MODEL_PROVIDERS.keys())}")
    
    # 1. Hugging Face Serverless API (Using OpenAI Wrapper)
    if provider == "HuggingFace":
        return ChatOpenAI(
            model=model_name,
            api_key=os.environ["HF_TOKEN"],
            base_url="https://router.huggingface.co/v1",
            **config
        )
    
    # 2. Google Generative AI
    if provider.__name__ == "ChatGoogleGenerativeAI":
        # Google expects 'model' and 'google_api_key'
        return provider(
            model=model_name,
            google_api_key=os.environ["GEMINI_API_KEY"],
            thinking_budget=512,
            **config
        )
        
    # 3. DeepInfra (Default Fallback)
    if api_key == os.environ.get("DEEPINFRA_API_KEY"):
        return ChatOpenAI(
            model=model_name,
            api_key=os.environ["DEEPINFRA_API_KEY"],
            base_url="https://api.deepinfra.com/v1/openai",
            **config # Make sure config is passed here too!
        )
    
    if api_key == os.environ["OPENROUTER_API_KEY"]:
        return ChatOpenAI(
            model=model_name,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )

    # 4. Catch-all for any other provider
    return provider(model_name=model_name, api_key=api_key, **config)


def write_to_file(path: Path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if the content is a pandas DataFrame first
    if isinstance(content, pd.DataFrame):
        # Let pandas handle the file writing directly; it's safer and cleaner
        content.to_csv(path, index=False, encoding="utf-8")
        return # Exit early since pandas handled the writing

    # Fallback for your original logic
    if isinstance(content, list):
        content = "\n".join(str(item) for item in content)
    elif not isinstance(content, str):
        content = str(content)
        
    path.write_text(content, encoding="utf-8")