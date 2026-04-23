from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_groq import ChatGroq
# ============================================================
# Model registry
# ============================================================

MODEL_PROVIDERS = {
    # OpenAI
    "gpt-4o": ChatOpenAI,
    "gpt-4o-mini": ChatOpenAI,
    "gpt-4-turbo": ChatOpenAI,
    "gpt-5-nano": ChatOpenAI,
    "deepseek-chat": ChatOpenAI,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": ChatOpenAI,
    "meta-llama/Llama-3.3-70B-Instruct": ChatOpenAI,
    # "llama-3.1-8b-instant": ChatGroq,
    # "llama-3.3-70b-versatile": ChatGroq,
    # Google
    "gemini-2.5-flash": ChatGoogleGenerativeAI,
    "gemini-1.5-pro": ChatGoogleGenerativeAI,
    "google/gemma-4-31B-it": ChatOpenAI,
    "google/gemma-3-12b-it": ChatOpenAI,
    "google/gemma-3-4b-it": ChatOpenAI,
}

LLM_BASE_CONFIG = {
    "temperature": 0.7,
}

VILLAGER_MODEL = "google/gemma-3-4b-it"  # small — villager side
WOLF_MODEL = "google/gemma-4-31B-it"  # large — wolf side

# ============================================================
# Experiment scenarios
# ============================================================

SCENARIO_CONFIG = {
    "baseline": {
        "coaching": False,
        "self_analyze": True,
        "deviation": False,
        "personality": False,
    },
    "coach_and_self_analyze": {
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
