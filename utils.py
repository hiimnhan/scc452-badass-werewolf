from __future__ import annotations # MUST be the absolute first line!
from typing import TYPE_CHECKING
from langchain_core.language_models import BaseChatModel
from pathlib import Path
from config import LLM_BASE_CONFIG, MODEL_PROVIDERS

if TYPE_CHECKING:
    from game import GameState


def get_llm(model_name: str, **kwargs) -> BaseChatModel:
    provider = MODEL_PROVIDERS.get(model_name)
    if not provider:
        raise ValueError(f"Model '{model_name}' is not supported. Choose from: {list(MODEL_PROVIDERS.keys())}")

    return provider(model_name=model_name, **{**LLM_BASE_CONFIG, **kwargs})


def write_to_file(path: Path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def log_game_summary(state: GameState, announcement):
    state._summary_logs.append(f"Round {state._round_num} - Phase {state._phase}: {announcement}")
    return state
