import json
from pathlib import Path
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from constants import (
    COACH_FEEDBACK_FILENAME,
    COACH_STRATEGY_FILENAME,
    COACH_DIRECTIVES_FILENAME,
    DIRECTIVES_UPTAKE_FILENAME,
    MAX_RETRIES,
)
from utils import write_to_file
import re


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

    def __init__(self, model: BaseChatModel, game_id: str = "", scenario: str = "baseline") -> None:
        self._model = model
        self._game_id = game_id
        self._scenario = scenario
        self._strategy = self._load_coach_strategy()

    # ── File path helpers ───────────────────────────────────────────────

    def _base_dir(self) -> Path:
        return Path(__file__).parent.parent

    @property
    def _coach_strategy_path(self) -> Path:
        return (self._base_dir() / "strategies" / self._scenario / COACH_STRATEGY_FILENAME).resolve()

    @property
    def _coach_feedback_path(self) -> Path:
        return (
            self._base_dir() / "game_logs" / self._scenario / f"game_{self._game_id}" / COACH_FEEDBACK_FILENAME
        ).resolve()

    @property
    def _coach_directives_path(self) -> Path:
        return (
            self._base_dir() / "game_logs" / self._scenario / f"game_{self._game_id}" / COACH_DIRECTIVES_FILENAME
        ).resolve()

    @property
    def _directives_uptake_path(self) -> Path:
        return (
            self._base_dir() / "game_logs" / self._scenario / f"game_{self._game_id}" / DIRECTIVES_UPTAKE_FILENAME
        ).resolve()

    def _previous_directives_path(self) -> Path | None:
        """Path to the directives file from the previous game, or None if none exists."""
        try:
            n = int(self._game_id)
        except (ValueError, TypeError):
            return None
        if n <= 1:
            return None
        prev = f"{n - 1:03d}"
        path = (
            self._base_dir() / "game_logs" / self._scenario / f"game_{prev}" / COACH_DIRECTIVES_FILENAME
        ).resolve()
        return path if path.exists() else None

    # ── Disk helpers ────────────────────────────────────────────────────

    def _load_coach_strategy(self) -> str:
        """Read the coach's own strategy from disk.
        Returns empty string on first game (file does not exist yet).
        """
        path = self._coach_strategy_path
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

    def _call_model(self, system: str, prompt: str, max_tokens: int = 500, timeout: int = 200) -> dict:
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
        if isinstance(resp, list):
            resp = " ".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in resp).strip()
        elif isinstance(resp, str):
            resp = resp.strip()

        result: dict = {}
        # 1. Clean up the response to extract just the JSON block
        clean_resp = resp.strip()

        # This regex looks for everything from the first '{' to the last '}'
        # re.DOTALL allows it to match across multiple lines
        match = re.search(r"(\{.*\})", clean_resp, re.DOTALL)

        if match:
            clean_resp = match.group(1)

        # 2. Attempt to parse the cleaned string
        try:
            result = json.loads(clean_resp)
        except json.JSONDecodeError:
            # Fallback if it is still irreparably broken
            result = {"raw": resp}

        result.setdefault("_prompt", prompt)
        return result

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
        # Step 1: Evaluate whether the PREVIOUS game's directives were taken up
        #         in THIS game. Skipped silently when no previous directives exist.
        #         Produces directives_uptake.json. Players NEVER read this file.
        self._evaluate_directives_uptake(game_record)

        # Step 2: Coach reviews the game and updates its own strategy
        self._update_coach_strategy(game_record)

        # Step 3: Coach uses its freshly updated strategy to write their feedback to the players
        self._generate_coach_feedback(game_record)

        # Step 4: Emit this game's structured directives for the NEXT game's
        #         evaluator. Evaluation-only artifact; no player reads it.
        self._generate_structured_directives(game_record)

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
  "strategy": "updated coaching rules as bullet points (<=500 words)",
  "reasoning": "what changed in your coaching approach and why (<=250 words)"
}}
No extra text, no markdown, no code fences.
"""

        for attempt in range(MAX_RETRIES):
            resp = self._call_model(system, prompt, max_tokens=2000)

            # Safely extract and clean the strategy string
            extracted_strategy = resp.get("strategy", "").strip()

            # Check if we actually got meaningful text back
            if extracted_strategy:
                new_strategy = extracted_strategy
                break  # We successfully got the strategy, exit the loop!
            else:
                print(f"Warning: Coach failed to generate a valid strategy JSON (Attempt {attempt + 1}/{MAX_RETRIES})")

        # Only overwrite the strategy file on the hard drive if we got a valid update
        if new_strategy:
            self._write_coach_strategy(new_strategy)
        else:
            raise ValueError(
                "Error: Coach completely failed to update strategy. Keeping the previous strategy for the next game."
            )

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

Write concise, actionable feedback for the villager-side players USING THE ROLES of the players, NOT their names.

Focus on:
- Specific mistakes made that allowed wolves to survive or mislead
- Deception signals that were present but missed
- Voting errors and when to override gut feelings with evidence
- Coordination failures between special roles (Seer, Guard, Witch)

This feedback will be read by each villager-side player before their next game and will shape their strategy update. Make it direct and specific — not generic advice.

Respond with ONLY a JSON object:
{{
  "feedback": "actionable feedback for villager players (<=500 words)",
  "key_mistakes": "the critical errors made this game (<=250 words)"
}}
No extra text, no markdown, no code fences.
"""
        key_mistakes = ""
        feedback = ""

        for attempt in range(MAX_RETRIES):
            resp = self._call_model(system, prompt, max_tokens=2000)

            # Safely extract and clean the text
            extracted_mistakes = resp.get("key_mistakes", "").strip()
            extracted_feedback = resp.get("feedback", "").strip()

            # We want the Coach to provide BOTH parts for a complete analysis
            if extracted_mistakes and extracted_feedback:
                key_mistakes = extracted_mistakes
                feedback = extracted_feedback
                break  # We successfully got both fields, exit the loop!
            else:
                print(f"Warning: Coach failed to generate full feedback JSON (Attempt {attempt + 1}/{MAX_RETRIES})")

        # Combine both fields into one readable feedback file
        feedback_parts: list[str] = []

        if key_mistakes:
            feedback_parts.append(f"KEY MISTAKES THIS GAME:\n{key_mistakes}")

        if feedback:
            feedback_parts.append(f"COACHING ADVICE FOR NEXT GAME:\n{feedback}")

        # Only write to the file if we actually have content
        if feedback_parts:
            self._write_coach_feedback("\n\n".join(feedback_parts))
        else:
            raise ValueError("Error: Coach completely failed to generate feedback.")

    # ── Directives uptake evaluation (measurement-only) ─────────────────

    def _generate_structured_directives(self, game_record: str) -> None:
        """Emit per-role Strategic Directive Sets (SDS) as JSON for this game.

        This file is NEVER read by any player. Its sole purpose is to serve as
        ground-truth against which the NEXT game's coach evaluates villager-side
        uptake of natural-language coaching.
        """
        system = (
            "You are an expert Werewolf coach producing structured strategic "
            "directives for villager-side roles (Villager, Seer, Guard, Witch). "
            "These directives will later be used to evaluate whether the villager "
            "side's behaviour in the next game is consistent with the coaching "
            "signal. You are NOT writing directives for the players to read — "
            "you are writing a ground-truth record of what 'good behaviour' "
            "looked like after this game."
        )

        sources: list[str] = [f"=== Full Game Record ===\n{game_record}"]
        if self._strategy:
            sources.append(f"=== Your Coaching Strategy ===\n{self._strategy}")
        context = "\n\n".join(sources)

        prompt = f"""
{context}

Produce a Strategic Directive Set (SDS) for each villager-side role.

STRICT RULES FOR EVERY DIRECTIVE:
1. ATOMIC: one trigger, one action, one phase. No compound rules.
2. OBSERVABLE TRIGGERS ONLY: round_num, alive_count, confirmed_wolves_count,
   votes_received_last_round, was_attacked_last_night, potions_remaining, etc.
   Never use subjective triggers like "when suspicious" or "if evidence is strong".
3. EXECUTABLE ACTIONS: a concrete verb the role can perform
   (VOTE_X, REVEAL, PROTECT_Y, POISON_Z, STAY_SILENT, ACCUSE_X).
4. If a directive requires information the role cannot observe, DO NOT EMIT IT.
   Example: Guard cannot identify the Seer, so "PROTECT_SEER" is not valid.
5. Phases are exactly one of: NIGHT, DEBATE, VOTE.

Respond with ONLY a JSON object using this exact shape:
{{
  "Seer":     [ {{ "id": "SEER-001", "phase": "...", "trigger": "...", "action": "...", "rationale_short": "...", "confidence": 0.0-1.0 }} ],
  "Guard":    [ ... ],
  "Witch":    [ ... ],
  "Villager": [ ... ]
}}
Limit 3-6 directives per role. No extra text, no markdown, no code fences.
"""

        directives = None
        for attempt in range(MAX_RETRIES):
            resp = self._call_model(system, prompt, max_tokens=2000)
            candidate = {k: v for k, v in resp.items() if k in ("Seer", "Guard", "Witch", "Villager")}
            if candidate and all(isinstance(v, list) for v in candidate.values()):
                directives = candidate
                break
            print(f"Warning: Coach failed to generate valid directives JSON (Attempt {attempt + 1}/{MAX_RETRIES})")

        if directives:
            write_to_file(self._coach_directives_path, json.dumps(directives, indent=2))
        else:
            print("Warning: Coach failed to emit structured directives; skipping SDS for this game.")

    def _evaluate_directives_uptake(self, game_record: str) -> None:
        """Evaluate whether the PREVIOUS game's directives were followed by the
        villager side in THIS game. Writes directives_uptake.json.

        Players never see this file. It exists purely so we can measure uptake of
        natural-language coaching without leaking structured cues into the player
        decision-time prompt pipeline.
        """
        prev_path = self._previous_directives_path()
        if prev_path is None:
            return  # game 1, or previous emission failed — nothing to evaluate

        try:
            previous_directives = json.loads(prev_path.read_text())
        except Exception as e:
            print(f"Warning: failed to load previous directives from {prev_path}: {e}")
            return

        flat_directives = []
        for role, directives in previous_directives.items():
            if not isinstance(directives, list):
                continue
            for d in directives:
                flat_directives.append({"role": role, **d})
        if not flat_directives:
            return

        system = (
            "You are a Werewolf game evaluator. You are given (a) a set of "
            "strategic directives produced at the end of the previous game for "
            "villager-side roles, and (b) the full public record of the game that "
            "was just played. For each directive, judge whether the triggering "
            "situation arose in the new game and what the player in that role "
            "actually did. You are observing and labeling, not coaching. Be "
            "precise, concrete, and honest. Cite round numbers. Do not invent "
            "events that are not in the game record."
        )

        prompt = f"""
=== Previous Game's Directives ===
{json.dumps(flat_directives, indent=2)}

=== This Game's Full Record ===
{game_record}

For EACH directive above, produce one evaluation entry. Return a JSON object
with this exact shape:

{{
  "evaluations": [
    {{
      "directive_id": "<copy from input>",
      "role": "<copy from input>",
      "directive_phase": "<copy from input>",
      "directive_trigger": "<copy from input>",
      "directive_action": "<copy from input>",
      "directive_rationale": "<copy rationale_short from input, or empty string>",
      "situation_arose": true | false,
      "arose_at_round": <int or null>,
      "arose_at_phase": "NIGHT" | "DEBATE" | "VOTE" | null,
      "actual_player_behavior": "<full natural-language description, no length limit, or null if situation did not arise>",
      "verdict": "FOLLOWED" | "IGNORED" | "PARTIALLY_FOLLOWED" | "NOT_APPLICABLE",
      "verdict_notes": "<full natural-language justification, no length limit>"
    }},
    ...
  ]
}}

Verdict rules:
- situation_arose=false -> verdict="NOT_APPLICABLE", arose_at_round=null, arose_at_phase=null, actual_player_behavior=null.
- situation_arose=true and action matched the directive -> "FOLLOWED".
- situation_arose=true and action clearly contradicted the directive -> "IGNORED".
- situation_arose=true and action was consistent with the directive's intent but deviated from its literal prescription -> "PARTIALLY_FOLLOWED" (explain the distinction in verdict_notes).
- Describe only observable behaviour. Do not speculate about the player's internal reasoning.

No markdown, no code fences, only the JSON object.
"""

        evaluations = None
        for attempt in range(MAX_RETRIES):
            resp = self._call_model(system, prompt, max_tokens=4000)
            candidate = resp.get("evaluations")
            if isinstance(candidate, list) and candidate:
                evaluations = candidate
                break
            print(f"Warning: Coach failed to produce valid uptake evaluations (Attempt {attempt + 1}/{MAX_RETRIES})")

        if evaluations is None:
            print("Warning: uptake evaluation failed for this game; skipping write.")
            return

        by_role: dict[str, list] = {}
        for ev in evaluations:
            role = ev.pop("role", "Unknown")
            by_role.setdefault(role, []).append(ev)

        try:
            prev_game_id = f"{int(self._game_id) - 1:03d}"
        except (ValueError, TypeError):
            prev_game_id = "unknown"

        output = {
            "evaluated_game_id": self._game_id,
            "directives_source_game_id": prev_game_id,
            "scenario": self._scenario,
            "by_role": by_role,
        }

        write_to_file(self._directives_uptake_path, json.dumps(output, indent=2))
