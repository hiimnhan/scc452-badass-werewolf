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

SCENARIO_CONFIG = {
    "baseline": {
        "coaching": False,
        "self_analyze": True,
        "deviation": False,
        "personality": False,
    },
    "coach_no_self_analyze": {
        "coaching": True,
        "self_analyze": False,
        "deviation": False,
        "personality": False,
    },
    "coach_and_self_analyze": {
        "coaching": True,
        "self_analyze": True,
        "deviation": False,
        "personality": False,
    },
    "self_analyze": {
        "coaching": True,
        "self_analyze": True,
        "deviation": False,
        "personality": False,
    },
    "deviate": {
        "coaching": True,
        "self_analyze": True,
        "deviation": True,
        "personality": False,
    },
    "personality": {
        "coaching": True,
        "self_analyze": True,
        "deviation": False,
        "personality": True,
    },
}

# ==========
