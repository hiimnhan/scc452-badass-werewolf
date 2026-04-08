import json
import sys
import types

# Stub langchain_core for this lightweight smoke test environment.
langchain_core = types.ModuleType("langchain_core")
language_models = types.ModuleType("langchain_core.language_models")


class BaseChatModel:
    pass


language_models.BaseChatModel = BaseChatModel
langchain_core.language_models = language_models
sys.modules.setdefault("langchain_core", langchain_core)
sys.modules.setdefault("langchain_core.language_models", language_models)

from players.guard import Guard


class Response:
    pass


class DummyModel:
    def invoke(self, prompt, max_tokens=0, timeout=0):
        r = Response()
        r.content = json.dumps({"target": "Alice", "is_deceptive": False, "analysis": "test"})
        return r


def test_guard_protect_tracks_last_guarded_player():
    g = Guard("Bob", DummyModel())
    t1, r1 = g.protect(["Alice", "Charlie"])
    g1 = g.last_guarded_player
    t2, r2 = g.protect(["Alice", "Charlie"])
    assert t1 == "Alice"
    assert g1 == t1
    assert isinstance(r1, dict)
    assert g.last_guarded_player == t2
    assert isinstance(r2, dict)
    assert "fallback" in r2 or r2.get("fallback") is None
