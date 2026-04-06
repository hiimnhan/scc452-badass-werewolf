# LLM Model Configuration

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# LLM Model Configuration =========
MODEL_PROVIDERS = {
    # OpenAI
    "gpt-4o": ChatOpenAI,
    "gpt-4o-mini": ChatOpenAI,
    "gpt-4-turbo": ChatOpenAI,
    # Google
    "gemini-2.5-flash": ChatGoogleGenerativeAI,
    "gemini-1.5-pro": ChatGoogleGenerativeAI,
}

LLM_BASE_CONFIG = {
    "temperature": 0.7,
}

# ==========
