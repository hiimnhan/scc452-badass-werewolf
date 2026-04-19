from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# ============================================================
# Model registry
# ============================================================

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

VILLAGER_MODEL = "gpt-4o-mini"  # small — villager side
WOLF_MODEL = "gpt-4o"  # large — wolf side

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
