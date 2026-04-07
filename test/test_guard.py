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


def main():
    g = Guard("Bob", DummyModel())
    t1, r1 = g.protect(["Alice", "Charlie"])
    g1 = g.last_guarded_player

    t2, r2 = g.protect(["Alice", "Charlie"])
    print("t1:", t1, "last:", g1)
    print("t2:", t2, "last:", g.last_guarded_player)
    print("fallback2:", r2.get("fallback"))


if __name__ == "__main__":
    main()
