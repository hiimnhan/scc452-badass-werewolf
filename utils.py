from langchain_core.language_models import BaseChatModel
from datetime import datetime
from typing import Dict, Optional
from config import LLM_BASE_CONFIG, MODEL_PROVIDERS


def get_llm(model_name: str, **kwargs) -> BaseChatModel:
    provider = MODEL_PROVIDERS.get(model_name)
    if not provider:
        raise ValueError(
            f"Model '{model_name}' is not supported. Choose from: {list(MODEL_PROVIDERS.keys())}"
        )

    return provider(model_name=model_name, **{**LLM_BASE_CONFIG, **kwargs})


def log_game_summary(state, announcement):
    state._summary_logs.append(
        f"Round {state._round_num} - Phase {state._phase}: {announcement}"
    )
    return state