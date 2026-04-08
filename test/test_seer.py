from players.seer import Seer
import json
import sys
import types
import traceback

# Stub langchain_core (same pattern as test_guard.py)
langchain_core = types.ModuleType("langchain_core")
language_models = types.ModuleType("langchain_core.language_models")


class BaseChatModel:
    pass


language_models.BaseChatModel = BaseChatModel
langchain_core.language_models = language_models
sys.modules.setdefault("langchain_core", langchain_core)
sys.modules.setdefault("langchain_core.language_models", language_models)


# DummyModel

def _model(content: dict):
    """Return a model that always responds with content serialised as JSON."""
    class _R:
        pass

    class _DM:
        def invoke(self, prompt, **kwargs):
            r = _R()
            r.content = json.dumps(content)
            return r
    return _DM()


# Reusable default models
DEFAULT_MODEL = _model({"target": "Alice", "analysis": "test"})
VOTE_MODEL = _model({"vote": "Bob",   "analysis": "test", "reasoning": "test"})
DEBATE_MODEL = _model({"statement": "I suspect Alice.", "is_deceptive": False,
                       "analysis": "test"})

# Models that return specific suspicion_level values for detect_deception tests
LOW_SUSPICION_MODEL = _model({
    "chain_of_thought": "seems fine", "is_deceptive": 0, "confidence": 0.5,
    "deception_type": "none", "reasoning": "no issues", "suspicion_level": 0.2,
})
HIGH_SUSPICION_MODEL = _model({
    "chain_of_thought": "very suspicious", "is_deceptive": 1, "confidence": 0.9,
    "deception_type": "fabrication", "reasoning": "lying", "suspicion_level": 0.95,
})
MID_SUSPICION_MODEL = _model({
    "chain_of_thought": "uncertain", "is_deceptive": 1, "confidence": 0.7,
    "deception_type": "misdirection", "reasoning": "diverting", "suspicion_level": 0.75,
})
REVEAL_DEBATE_MODEL = _model({
    "statement": "I am the Seer and I investigated Alice — she is a wolf!",
    "is_deceptive": False, "analysis": "revealing now",
})


# unmask()

def test_unmask_returns_valid_target():
    s = Seer("Charlie", DEFAULT_MODEL)
    target, log = s.unmask(["Alice", "Bob", "Dave"])
    assert target == "Alice"
    assert isinstance(log, dict)
    assert "error" not in log


def test_unmask_excludes_self():
    """Seer must never target itself — fallback picks first valid available target."""
    s = Seer("Alice", DEFAULT_MODEL)    # model returns "Alice" but self is "Alice"
    target, log = s.unmask(["Alice", "Bob"])
    assert target == "Bob"              # "Alice" excluded, fallback → "Bob"


def test_unmask_soft_preference_for_uninvestigated():
    """
    target_pool shown to LLM prefers uninvestigated players, but validation
    accepts ANY player in available_targets. The preference is soft — the LLM
    may still pick an already-investigated player and it will be accepted.
    """
    s = Seer("Charlie", DEFAULT_MODEL)  # model always returns "Alice"
    s.reveal_and_update("Alice", is_wolf=False)
    # Alice is investigated but still in available_targets — model's choice accepted
    target, log = s.unmask(["Alice", "Bob", "Dave"])
    assert target == "Alice"
    assert "fallback" not in log


def test_unmask_fallback_all_investigated():
    """If all alive players are investigated, target_pool = available_targets — no error."""
    s = Seer("Charlie", DEFAULT_MODEL)  # model returns "Alice"
    s.reveal_and_update("Alice", is_wolf=False)
    s.reveal_and_update("Bob",   is_wolf=True)
    target, log = s.unmask(["Alice", "Bob"])
    assert target in ["Alice", "Bob"]
    assert "error" not in log


def test_unmask_adds_note():
    s = Seer("Charlie", DEFAULT_MODEL)
    s.unmask(["Alice", "Bob"])
    assert len(s._current_game_notes) == 1
    assert "investigation" in s._current_game_notes[0].lower()


def test_unmask_fallback_invalid_model_response():
    """Model returns a player not in available_players — use target_pool[0]."""
    bad_model = _model({"target": "NonExistentPlayer", "analysis": "bad"})
    s = Seer("Charlie", bad_model)
    target, log = s.unmask(["Alice", "Bob"])
    assert target in ["Alice", "Bob"]
    assert "fallback" in log


def test_unmask_error_on_none_alive_players():
    """Case 6: None must return error, not crash with TypeError."""
    s = Seer("Charlie", DEFAULT_MODEL)
    target, log = s.unmask(None)
    assert target == ""
    assert "error" in log


def test_unmask_error_on_empty_alive_players():
    s = Seer("Charlie", DEFAULT_MODEL)
    target, log = s.unmask([])
    assert target == ""
    assert "error" in log


def test_unmask_error_only_self_in_alive_players():
    s = Seer("Charlie", DEFAULT_MODEL)
    target, log = s.unmask(["Charlie"])
    assert target == ""
    assert "error" in log


def test_unmask_error_when_seer_is_dead():
    """Case 4: dead seer must not act."""
    s = Seer("Charlie", DEFAULT_MODEL)
    s._is_alive = False
    target, log = s.unmask(["Alice", "Bob"])
    assert target == ""
    assert "error" in log
    assert "dead" in log["error"].lower()

# reveal_and_update()


def test_reveal_wolf_sets_suspicion_and_note():
    s = Seer("Charlie", DEFAULT_MODEL)
    s.reveal_and_update("Alice", is_wolf=True)
    assert s._suspicions["Alice"] == 1.0
    assert s._investigations == [{"player": "Alice", "is_wolf": True}]
    assert any("wolf" in n.lower() for n in s._current_game_notes)


def test_reveal_innocent_sets_suspicion_and_note():
    s = Seer("Charlie", DEFAULT_MODEL)
    s.reveal_and_update("Bob", is_wolf=False)
    assert s._suspicions["Bob"] == 0.0
    assert s._investigations == [{"player": "Bob", "is_wolf": False}]
    assert any("not" in n.lower() for n in s._current_game_notes)


def test_reveal_idempotent_same_result():
    """Case 1: calling twice with the same result is a silent no-op."""
    s = Seer("Charlie", DEFAULT_MODEL)
    s.reveal_and_update("Alice", is_wolf=True)
    notes_before = len(s._current_game_notes)
    invs_before = len(s._investigations)
    # same result — no side effects
    s.reveal_and_update("Alice", is_wolf=True)
    assert len(s._investigations) == invs_before
    assert len(s._current_game_notes) == notes_before


def test_reveal_conflict_raises_value_error():
    """Case 1: conflicting results for same player must raise, not silently corrupt."""
    s = Seer("Charlie", DEFAULT_MODEL)
    s.reveal_and_update("Alice", is_wolf=True)
    try:
        s.reveal_and_update("Alice", is_wolf=False)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "conflicting" in str(e).lower()


def test_reveal_none_player_name_raises():
    """Case 8: None player_name must raise ValueError."""
    s = Seer("Charlie", DEFAULT_MODEL)
    try:
        s.reveal_and_update(None, is_wolf=True)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_reveal_empty_player_name_raises():
    """Case 8: empty string player_name must raise ValueError."""
    s = Seer("Charlie", DEFAULT_MODEL)
    try:
        s.reveal_and_update("", is_wolf=False)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_multiple_investigations_accumulated_correctly():
    s = Seer("Charlie", DEFAULT_MODEL)
    s.reveal_and_update("Alice", is_wolf=True)
    s.reveal_and_update("Bob",   is_wolf=False)
    assert len(s._investigations) == 2
    assert s._suspicions["Alice"] == 1.0
    assert s._suspicions["Bob"] == 0.0


# Helper methods

def test_get_confirmed_wolves_and_innocents():
    s = Seer("Charlie", DEFAULT_MODEL)
    s.reveal_and_update("Alice", is_wolf=True)
    s.reveal_and_update("Bob",   is_wolf=False)
    assert s.get_confirmed_wolves() == ["Alice"]
    assert s.get_confirmed_innocents() == ["Bob"]


def test_is_confirmed_returns_none_for_unknown():
    s = Seer("Charlie", DEFAULT_MODEL)
    assert s._is_confirmed("Dave") is None


def test_confirmed_label_wolf():
    s = Seer("Charlie", DEFAULT_MODEL)
    s.reveal_and_update("Alice", is_wolf=True)
    assert "CONFIRMED WOLF" in s._confirmed_label("Alice")


def test_confirmed_label_innocent():
    s = Seer("Charlie", DEFAULT_MODEL)
    s.reveal_and_update("Bob", is_wolf=False)
    assert "CONFIRMED INNOCENT" in s._confirmed_label("Bob")


def test_confirmed_label_uninvestigated_is_plain_name():
    s = Seer("Charlie", DEFAULT_MODEL)
    assert s._confirmed_label("Dave") == "Dave"


def test_format_investigation_results_empty():
    s = Seer("Charlie", DEFAULT_MODEL)
    result = s._format_investigation_results()
    assert "none" in result.lower() or "not investigated" in result.lower()


def test_format_investigation_results_populated():
    s = Seer("Charlie", DEFAULT_MODEL)
    s.reveal_and_update("Alice", is_wolf=True)
    s.reveal_and_update("Bob",   is_wolf=False)
    result = s._format_investigation_results()
    assert "Alice" in result
    assert "werewolf" in result.lower()
    assert "Bob" in result
    assert "not" in result.lower()

# detect_deception() override


def test_detect_deception_does_not_overwrite_confirmed_wolf():
    """Confirmed wolf suspicion stays at 1.0 regardless of LLM output."""
    s = Seer("Charlie", LOW_SUSPICION_MODEL)    # LLM would give 0.2
    s.reveal_and_update("Alice", is_wolf=True)
    s.detect_deception("Alice", "I am totally innocent.")
    assert s._suspicions["Alice"] == 1.0


def test_detect_deception_does_not_overwrite_confirmed_innocent():
    """Confirmed innocent suspicion stays at 0.0 regardless of LLM output."""
    s = Seer("Charlie", HIGH_SUSPICION_MODEL)   # LLM would give 0.95
    s.reveal_and_update("Bob", is_wolf=False)
    s.detect_deception("Bob", "Charlie is the wolf!")
    assert s._suspicions["Bob"] == 0.0


def test_detect_deception_confirmed_response_flags_correctly():
    """Response for confirmed player includes _confirmed_by_investigation=True."""
    s = Seer("Charlie", LOW_SUSPICION_MODEL)
    s.reveal_and_update("Alice", is_wolf=True)
    resp = s.detect_deception("Alice", "I am innocent.")
    assert resp.get("_confirmed_by_investigation") is True
    assert resp["suspicion_level"] == 1.0


def test_detect_deception_preserves_original_llm_suspicion_field():
    """_original_llm_suspicion captures what the LLM actually returned."""
    s = Seer("Charlie", LOW_SUSPICION_MODEL)    # LLM returns 0.2
    s.reveal_and_update("Alice", is_wolf=True)
    resp = s.detect_deception("Alice", "Trust me.")
    # The LLM returned 0.2 — preserved separately even though fixed score is 1.0
    assert resp.get("_original_llm_suspicion") == 0.2


def test_detect_deception_updates_freely_for_uninvestigated():
    """For players not yet investigated, the suspicion score updates normally."""
    s = Seer("Charlie", MID_SUSPICION_MODEL)    # LLM returns 0.75
    s.detect_deception("Dave", "I suspect Charlie.")
    assert abs(s._suspicions["Dave"] - 0.75) < 1e-6

# vote() — LLM has full freedom (no forced vote)


def test_vote_llm_has_freedom_with_confirmed_wolf():
    """
    No forced vote. LLM returns 'Bob' even though Alice is confirmed wolf.
    This is valid — seer hiding identity is realistic strategic play.
    """
    s = Seer("Charlie", VOTE_MODEL)             # model returns "Bob"
    s.reveal_and_update("Alice", is_wolf=True)
    target, log = s.vote(["Alice", "Bob", "Dave"])
    assert target == "Bob"                      # accepted — no forced override
    assert "_forced_by_investigation" not in log


def test_vote_valid_target_returned():
    """vote() returns a target from alive_players in all cases."""
    s = Seer("Charlie", VOTE_MODEL)
    s.reveal_and_update("Alice", is_wolf=True)
    s.reveal_and_update("Bob",   is_wolf=False)
    target, log = s.vote(["Alice", "Bob", "Dave"])
    assert target in ["Alice", "Bob", "Dave"]
    assert "error" not in log


def test_vote_works_with_no_investigations():
    s = Seer("Charlie", VOTE_MODEL)
    target, log = s.vote(["Alice", "Bob", "Dave"])
    assert target in ["Alice", "Bob", "Dave"]
    assert "error" not in log


def test_vote_adds_note():
    s = Seer("Charlie", VOTE_MODEL)
    s.vote(["Alice", "Bob"])
    assert any("voted" in n.lower() for n in s._current_game_notes)


def test_vote_error_no_targets():
    s = Seer("Charlie", VOTE_MODEL)
    target, log = s.vote([])
    assert target == ""
    assert "error" in log


def test_vote_error_only_self_in_alive():
    s = Seer("Charlie", VOTE_MODEL)
    target, log = s.vote(["Charlie"])
    assert target == ""
    assert "error" in log

# debate() — LLM has full freedom; _role_revealed tracked


def test_debate_returns_statement():
    s = Seer("Charlie", DEBATE_MODEL)
    statement, log = s.debate([["Alice", "I think Bob is suspicious."]])
    assert statement == "I suspect Alice."
    assert "error" not in log


def test_debate_llm_can_withhold_investigation_knowledge():
    """
    LLM is not forced to mention confirmed results.
    Plain statement with no wolf mention is valid — seer hiding identity.
    """
    s = Seer("Charlie", DEBATE_MODEL)           # returns "I suspect Alice."
    s.reveal_and_update("Bob", is_wolf=True)
    statement, log = s.debate([])
    assert statement == "I suspect Alice."      # no mention of Bob — valid
    assert "error" not in log


def test_debate_tracks_role_revealed_flag():
    """_role_revealed becomes True when statement contains seer keywords."""
    s = Seer("Charlie", REVEAL_DEBATE_MODEL)
    assert s._role_revealed is False
    s.debate([])
    assert s._role_revealed is True


def test_debate_does_not_set_flag_for_ordinary_statement():
    # "I suspect Alice." — no seer keyword
    s = Seer("Charlie", DEBATE_MODEL)
    s.debate([])
    assert s._role_revealed is False


def test_debate_flag_persists_across_subsequent_calls():
    """Once revealed, _role_revealed stays True — not reset between turns."""
    s = Seer("Charlie", REVEAL_DEBATE_MODEL)
    s.debate([])
    assert s._role_revealed is True
    # Switch to a non-revealing model for the second call
    s._model = DEBATE_MODEL.__class__()         # won't match — use the factory
    s._model = _model({"statement": "Bob is suspicious.", "is_deceptive": False,
                       "analysis": "following up"})
    s.debate([["Alice", "I agree."]])
    assert s._role_revealed is True             # still True, not reset


def test_debate_adds_note():
    s = Seer("Charlie", DEBATE_MODEL)
    s.debate([])
    assert any("debate" in n.lower() for n in s._current_game_notes)


def test_debate_error_on_empty_model_response():
    """Model returning no statement should return error, not crash."""
    empty_model = _model(
        {"statement": "", "is_deceptive": False, "analysis": ""})
    s = Seer("Charlie", empty_model)
    statement, log = s.debate([])
    assert statement == "" or "error" in log


# repr

def test_seer_repr():
    s = Seer("Charlie", DEFAULT_MODEL)
    r = repr(s)
    assert "Charlie" in r
    assert "Seer" in r


if __name__ == "__main__":
    tests = [
        # unmask
        test_unmask_returns_valid_target,
        test_unmask_excludes_self,
        test_unmask_soft_preference_for_uninvestigated,
        test_unmask_fallback_all_investigated,
        test_unmask_adds_note,
        test_unmask_fallback_invalid_model_response,
        test_unmask_error_on_none_alive_players,
        test_unmask_error_on_empty_alive_players,
        test_unmask_error_only_self_in_alive_players,
        test_unmask_error_when_seer_is_dead,
        # reveal_and_update
        test_reveal_wolf_sets_suspicion_and_note,
        test_reveal_innocent_sets_suspicion_and_note,
        test_reveal_idempotent_same_result,
        test_reveal_conflict_raises_value_error,
        test_reveal_none_player_name_raises,
        test_reveal_empty_player_name_raises,
        test_multiple_investigations_accumulated_correctly,
        # helpers
        test_get_confirmed_wolves_and_innocents,
        test_is_confirmed_returns_none_for_unknown,
        test_confirmed_label_wolf,
        test_confirmed_label_innocent,
        test_confirmed_label_uninvestigated_is_plain_name,
        test_format_investigation_results_empty,
        test_format_investigation_results_populated,
        # detect_deception
        test_detect_deception_does_not_overwrite_confirmed_wolf,
        test_detect_deception_does_not_overwrite_confirmed_innocent,
        test_detect_deception_confirmed_response_flags_correctly,
        test_detect_deception_preserves_original_llm_suspicion_field,
        test_detect_deception_updates_freely_for_uninvestigated,
        # vote
        test_vote_llm_has_freedom_with_confirmed_wolf,
        test_vote_valid_target_returned,
        test_vote_works_with_no_investigations,
        test_vote_adds_note,
        test_vote_error_no_targets,
        test_vote_error_only_self_in_alive,
        # debate
        test_debate_returns_statement,
        test_debate_llm_can_withhold_investigation_knowledge,
        test_debate_tracks_role_revealed_flag,
        test_debate_does_not_set_flag_for_ordinary_statement,
        test_debate_flag_persists_across_subsequent_calls,
        test_debate_adds_note,
        test_debate_error_on_empty_model_response,
        # repr
        test_seer_repr,
    ]

    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
