from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

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
    "o3-mini": ChatOpenAI,
    "openai/gpt-oss-120b": ChatOpenAI,
    "openai/gpt-oss-20b": ChatOpenAI,
}

LLM_BASE_CONFIG = {
    "temperature": 0.7,
}

# ============================================================
# Experiment scenarios
# ============================================================

SCENARIO_CONFIG = {
    # "baseline_12b_12b": {
    #     "villager_model": "google/gemma-3-12b-it",
    #     "wolf_model": "google/gemma-3-12b-it",
    #     "coaching": False,
    #     "self_analyze": True,
    # },
    # "coach_12b_12b": {
    #     "villager_model": "google/gemma-3-12b-it",
    #     "wolf_model": "google/gemma-3-12b-it",
    #     "coaching": True,
    #     "self_analyze": True,
    # },
    "baseline_12b_31B": {
        "villager_model": "google/gemma-3-12b-it",
        "wolf_model": "google/gemma-4-31B-it",
        "coaching": False,
        "self_analyze": True,
    },
    "coach_12b_31B": {
        "villager_model": "google/gemma-3-12b-it",
        "wolf_model": "google/gemma-4-31B-it",
        "coaching": True,
        "self_analyze": True,
    },
    "baseline_31B_31B": {
        "villager_model": "google/gemma-4-31B-it",
        "wolf_model": "google/gemma-4-31B-it",
        "coaching": False,
        "self_analyze": True,
    },
    "coach_31B_31B": {
        "villager_model": "google/gemma-4-31B-it",
        "wolf_model": "google/gemma-4-31B-it",
        "coaching": True,
        "self_analyze": True,
    },
}
