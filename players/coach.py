import json
from pathlib import Path
from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from constants import COACH_FEEDBACK_FILENAME, COACH_STRATEGY_FILENAME
from utils import write_to_file

class Coach:
    """
    Post-game agent that reviews the full game and produces two outputs:

      1. Updates its own coaching strategy (COACH_STRATEGY_FILENAME): "How should I coach villager-side players better next time?"

      2. Writes player-facing feedback (COACH_FEEDBACK_FILENAME): "What should villager-side players do differently next time?"
         This file is read by BasePlayer._coach_feedback.

    The Coach is NOT a BasePlayer — it has no role, no suspicion scores, and takes no in-game actions. It only runs post-game when coaching=True.

    ── File layout ─────────────────────────────────────────────────────────
      strategies/  COACH_STRATEGY_FILENAME                   the coach's OWN accumulated strategy (how it coaches — persists across games)
      game_logs/ game_{game_id}/  {COACH_FEEDBACK_FILENAME}  feedback written FOR villager players for each game

    ── Usage in run.py ─────────────────────────────────────────────────────
      if coaching:
          coach.run(game_record)                   # step 1: coach updates itself
                                                   # step 2: coach writes feedback to the villager-side players
      for player in all_players:
          player._update_strategy(                 # villagers read the feedback file
              self_analyze=self_analyze,           # wolves ignore coaching entirely
              coaching=coaching,
          )

    Always call coach.run() BEFORE player._update_strategy() so the feedback file exists on disk when villager-side players go to read it.
    """

    def __init__(self, model: BaseChatModel, game_id: str = "") -> None:
        self._model = model
        self._game_id = game_id
        self._strategy = self._load_coach_strategy()
        
        # ── File path ───────────────────────────────────────────────────
        BASE_DIR_PATH = Path(__file__).parent.parent
        self._coach_strategy_path = (BASE_DIR_PATH / "strategies" / COACH_STRATEGY_FILENAME).resolve()
        self._coach_feedback_path = (BASE_DIR_PATH / "game_logs" / f"game_{self._game_id}" / COACH_FEEDBACK_FILENAME).resolve()

    # ── Disk helpers ────────────────────────────────────────────────────

    def _load_coach_strategy(self) -> str:
        """Read the coach's own strategy from disk.
        Returns empty string on first game (file does not exist yet).
        """
        path = self._coach_strategy_path()
        if not path.exists():
            return ""
        try:
            return path.read_text().strip()
        except Exception:
            return ""

    def _write_coach_strategy(self, strategy: str) -> None:
        """Persist the coach's updated strategy and refresh the in-memory copy."""
        write_to_file(self._coach_strategy_path, strategy)
        self._strategy = strategy

    def _write_coach_feedback(self, feedback: str) -> None:
        """Write coach feedback to the file BasePlayer._coach_feedback reads."""
        write_to_file(self._coach_feedback_path, feedback)

    # ── LLM call ────────────────────────────────────────────────────────

    def _call_model(self, system: str, prompt: str, max_tokens: int = 500, timeout: int = 15) -> dict:
        """Send a two-message request to the LLM.
        Returns parsed JSON dict; falls back to {"raw": str} on parse failure.
        """
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ]
        kwargs = {"timeout": timeout}

        # Fix for the LangChain/Google kwarg bug
        if "GoogleGenerativeAI" not in type(self._model).__name__:
            # Only enforce max_tokens at the invoke level for OpenAI/Anthropic.
            # Gemini will naturally stop based on the prompt's word limits!
            kwargs["max_tokens"] = max_tokens

        resp = self._model.invoke(messages, **kwargs).content
        resp = resp.strip() if isinstance(resp, str) else resp

        try:
            return json.loads(resp)
        except json.JSONDecodeError:
            return {"raw": resp}

    # ── Public API ──────────────────────────────────────────────────────

    def run(self, game_record: str) -> None:
        """Run the full post-game coaching pipeline.

        Step 1 — Coach self-update:
            Review what happened and whether prior coaching advice worked.
            Update coach_strategy.txt with lessons about how to coach better.

        Step 2 — Write coach feedback:
            Using the freshly updated coaching strategy, produce actionable feedback for villager-side players and write it to the feedback file.

        Args:
            game_record: The full public game record as a formatted string.
                         Produced by GameState (all public events in order).
                         Example format:
                           Round 1 Night: Dave was eliminated.
                           Round 1 Day: Bob accused Charlie. Charlie exiled.
                           Round 2 Night: No one died.
                           ...
        """
        # Step 1: Coach reviews the game and updates its own strategy first
        self._update_coach_strategy(game_record)

        # Step 2: Coach uses its freshly updated strategy to write their feedback to the players
        self._generate_coach_feedback(game_record)

    def _update_coach_strategy(self, game_record: str) -> None:
        """Review whether prior coaching advice helped villagers this game.
        Update coach_strategy.txt with lessons about more effective coaching.
        """
        system = "You are an expert Werewolf game coach. Your job is to help villager-side players (Villager, Seer, Guard, Witch) improve their deception-detection and coordination over multiple games. You do NOT coach Werewolves."

        sources: list[str] = [f"=== Full Game Record ===\n{game_record}"]
        if self._strategy:
            sources.append(f"=== Your Previous Coaching Strategy ===\n{self._strategy}")

        context = "\n\n".join(sources)

        prompt = f"""
{context}

Review the game above. Ask yourself:
- Did the villagers make the mistakes your previous strategy warned against?
- Did your prior coaching advice actually help, or was it ignored / ineffective?
- What patterns of villager failure appeared that your strategy did not address?
- What coaching approaches should you add, change, or drop?

Produce an updated coaching strategy — rules for how YOU should coach villager players in future games to help them win more effectively.

Respond with ONLY a JSON object:
{{
  "strategy": "updated coaching rules as bullet points (<=150 words)",
  "reasoning": "what changed in your coaching approach and why (<=40 words)"
}}
No extra text, no markdown, no code fences.
"""
        resp = self._call_model(system, prompt, max_tokens=600)
        new_strategy = resp.get("strategy", "")

        if new_strategy:
            self._write_coach_strategy(new_strategy)

    def _generate_coach_feedback(self, game_record: str) -> None:
        """Write actionable feedback for villager-side players.
        This is the file BasePlayer._coach_feedback reads.
        Uses the freshly updated coaching strategy from Step 1.
        """
        system = "You are an expert Werewolf game coach writing feedback for the villager-side players (Villager, Seer, Guard, Witch) after a completed game."

        sources: list[str] = [f"=== Full Game Record ===\n{game_record}"]
        if self._strategy:
            sources.append(f"=== Your Coaching Strategy ===\n{self._strategy}")

        context = "\n\n".join(sources)

        prompt = f"""
{context}

Write concise, actionable feedback for the villager-side players.

Focus on:
- Specific mistakes made that allowed wolves to survive or mislead
- Deception signals that were present but missed
- Voting errors and when to override gut feelings with evidence
- Coordination failures between special roles (Seer, Guard, Witch)

This feedback will be read by each villager-side player before their next game
and will shape their strategy update. Make it direct and specific — not generic advice.

Respond with ONLY a JSON object:
{{
  "feedback": "actionable feedback for villager players (<=150 words)",
  "key_mistakes": "the 2-3 most critical errors made this game (<=50 words)"
}}
No extra text, no markdown, no code fences.
"""
        resp = self._call_model(system, prompt, max_tokens=600)

        # Combine both fields into one readable feedback file
        feedback_parts: list[str] = []

        key_mistakes = resp.get("key_mistakes", "")
        if key_mistakes:
            feedback_parts.append(f"KEY MISTAKES THIS GAME:\n{key_mistakes}")

        feedback = resp.get("feedback", "")
        if feedback:
            feedback_parts.append(f"COACHING ADVICE FOR NEXT GAME:\n{feedback}")

        if feedback_parts:
            self._write_coach_feedback("\n\n".join(feedback_parts))