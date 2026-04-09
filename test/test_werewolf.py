import json
import sys
import types
import traceback

langchain_core = types.ModuleType("langchain_core")
language_models = types.ModuleType("langchain_core.language_models")
class BaseChatModel: pass
language_models.BaseChatModel = BaseChatModel
langchain_core.language_models = language_models
sys.modules.setdefault("langchain_core", langchain_core)
sys.modules.setdefault("langchain_core.language_models", language_models)

from players.wolf import Wolf

# fake model
def _mock_llm(content: dict):
    class _R: pass
    class _DM:
        def invoke(self, prompt, **kwargs): 
            r = _R()
            r.content = json.dumps(content) # Returns fake JSON
            return r
    return _DM()

# Test case
def test_wolf_eliminate_itself(): 
    """Wolf should never target themselves."""
    model = _mock_llm({"target": "Jin", "analysis": "suicide attempt"})
    w = Wolf("Jin", model)
    w.wolf_debate(alive_players=["Alice", "Bob", "Charlie", "Jin"], other_wolves=["Bob"], dialogue_history=[])
    target, log = w.eliminate(["Alice", "Bob", "Charlie", "Jin"])
    assert target != "Jin"
    assert target in ["Alice", "Bob", "Charlie"] 

def test_wolf_debate_adds_note():
    """Wolf should be able to add note."""
    model = _mock_llm({"statement": "Let's kill Alice", "analysis": "test"})
    w = Wolf("Jin", model)
    w.wolf_debate(["Alice", "Bob"], ["Bob"], [])
    last_note = w._current_game_notes[-1]
    print(f"Notes count: {len(w._current_game_notes)}")
    print(f"Content of last note: {last_note}")
    assert len(w._current_game_notes) > 0
    assert "Wolf Chat" in w._current_game_notes[-1]
    
def test_wolf_eliminate_its_friend():
    """Wolf should not eliminate its teammates."""
    model = _mock_llm({"target": "Bob", "analysis": "I'm confused"}) # Bob is werewolf in this case
    w = Wolf("Jin", model)
    w.wolf_debate(alive_players=["Alice", "Bob", "Charlie", "Jin"], other_wolves=["Bob"], dialogue_history=[])
    
    # Bob is a teammate, Alice, Charlie is a Villager
    target, log = w.eliminate(alive_players=["Alice", "Bob", "Charlie", "Jin"])
    
    assert target == "Alice"
    assert target != "Bob"

if __name__ == "__main__":
    tests = [test_wolf_eliminate_itself, test_wolf_debate_adds_note, test_wolf_eliminate_its_friend]
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
        except Exception as e:
            print(f"FAIL: {t.__name__}\n{traceback.format_exc()}")