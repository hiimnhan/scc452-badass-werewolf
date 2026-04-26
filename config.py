from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_groq import ChatGroq

# ============================================================
# Model registry
# ============================================================

MODEL_PROVIDERS = {
    # DeepSeek
    "deepseek-ai/DeepSeek-V4-Flash": ChatOpenAI,
    "deepseek-ai/DeepSeek-V3.2": ChatOpenAI,
    # Google
    "gemini-2.5-flash": ChatGoogleGenerativeAI, # Does not work well. The wolves keep asking the player who was targeted how they survived.
    "gemini-1.5-pro": ChatGoogleGenerativeAI,
    "google/gemma-4-26B-A4B-it": ChatOpenAI,
    "google/gemma-4-31B-it": ChatOpenAI,
    "google/gemma-3-27b-it": ChatOpenAI,
    "google/gemma-3-12b-it": ChatOpenAI,
    "google/gemma-3-4b-it": ChatOpenAI,
    "liquid/lfm-2-24b-a2b": ChatOpenAI,
    "zai-org/GLM-4.7-Flash": ChatOpenAI,
    "nvidia/nemotron-3-super-120b-a12b:free": ChatOpenAI,
}

LLM_BASE_CONFIG = {
    "temperature": 0.7,
}

# Models
VILLAGER_MODEL = "google/gemma-3-12b-it"  # small — villager side
WOLF_MODEL = "google/gemma-4-31B-it"  # large — wolf side

# "google/gemma-3-12b-it"
# VILLAGER_MODEL = "liquid/lfm-2-24b-a2b"  # small — villager side
# WOLF_MODEL = "google/gemma-4-31B-it"  # large — wolf side

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
    "coach_without_self_analyze": {
        "coaching": True,
        "self_analyze": False,
        "deviation": False,
        "personality": False,
    },
}
