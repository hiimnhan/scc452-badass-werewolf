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
}

LLM_BASE_CONFIG = {
    "temperature": 0.7,
}

VILLAGER_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # small — villager side
WOLF_MODEL = "meta-llama/Llama-3.3-70B-Instruct"  # large — wolf side

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
