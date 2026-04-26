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
         # Step 1: evaluate previous game's directives against this game's record
        self._evaluate_directives_uptake(game_record)

        # Step 2: coach updates its own meta-strategy
        self._update_coach_strategy(game_record)

        # Step 3: coach writes prose feedback for players (READ by villagers)
        self._generate_coach_feedback(game_record)

        # Step 4: coach parses its own feedback into structured directives
        #         (READ by next game's evaluator only). Must run AFTER step 3.
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
        """Parse the prose feedback just written for players into structured
        directives. Merges with any existing directives file for this game so
        that re-runs cannot lose data already captured.

        INPUT:  coach_feedback.txt (just written for players)
        OUTPUT: coach_directives.json with merged {by_role, coverage}

        The game_record argument is unused; directives must reflect ONLY what
        the players were told. Kept in signature for compatibility with run().
        """
        feedback_path = self._coach_feedback_path
        if not feedback_path.exists():
            print("Warning: coach_feedback.txt missing; cannot align directives. Skipping SDS for this game.")
            return

        try:
            feedback_text = feedback_path.read_text().strip()
        except Exception as e:
            print(f"Warning: failed to read coach feedback: {e}")
            return

        if not feedback_text:
            print("Warning: coach feedback is empty; nothing to convert into directives.")
            return

        system = (
            "You are a Werewolf coaching parser. You will be given the prose "
            "feedback that was just written for villager-side players. Convert "
            "EVERY distinct piece of advice in that feedback into a structured "
            "directive whenever you reasonably can. Bias toward emission: only "
            "drop advice that is genuinely un-checkable. You MUST NOT introduce "
            "advice that is not present in the feedback. You MUST NOT reword "
            "advice into something stronger or different than what was actually "
            "communicated. You are parsing, not inventing, but you should be "
            "thorough — the feedback is rich and most of it should become "
            "directives."
        )

        prompt = f"""
    === Coach Feedback Just Written for Players ===
    {feedback_text}

    Convert the feedback into structured directives. Be thorough — emit a
    directive for EVERY distinct piece of advice you can reasonably express
    as observable behaviour. There is no upper limit; if the feedback contains
    twelve pieces of advice for the Seer, emit twelve directives for the Seer.

    EMIT a directive when you can express the advice as:
    - one TRIGGER (a condition the role can recognize from observable game state)
    - one ACTION (a concrete behaviour the role can perform)
    - one PHASE (NIGHT, DEBATE, or VOTE)

    Triggers can reference any observable game state — round_num, alive_count,
    confirmed_wolves_count, votes_received_last_round, was_attacked_last_night,
    potions_remaining, exiled_player_was_wolf, debate_target_was_attacked, etc.
    Avoid only purely subjective triggers ("when the mood feels off").

    Actions can be ANY observable behaviour the role can take. Examples
    (non-exhaustive): VOTE_X, VOTE_AGAINST_X, ABSTAIN, REVEAL_ROLE, REVEAL_INVESTIGATION,
    PROTECT_X, ENDORSE_X, DEFEND_X, CHALLENGE_X, ACCUSE_X, WITHHOLD_INFORMATION,
    STAY_SILENT, INVESTIGATE_X, POISON_X, SAVE_X, REQUEST_VOTE_ON_X. Invent
    a clear action verb if none of these fit — the evaluator will interpret it.

    DROP a piece of advice ONLY when:
    - It is purely subjective and admits no observable interpretation, OR
    - It does not apply to a specific role, OR
    - The role genuinely cannot observe what the advice depends on (e.g. Guard
    cannot identify the Seer). Note: if the advice has any reasonable
    interpretation that IS observable, prefer to emit, not drop.

    Phases are exactly NIGHT, DEBATE, or VOTE.

    Respond with ONLY a JSON object using this exact shape:

    {{
    "by_role": {{
        "Seer":     [ {{ "id": "SEER-001", "phase": "...", "trigger": "...", "action": "...", "rationale_short": "verbatim or close paraphrase of the feedback line this came from", "confidence": 0.0-1.0 }} ],
        "Guard":    [ ... ],
        "Witch":    [ ... ],
        "Villager": [ ... ]
    }},
    "coverage": {{
        "total_advice_pieces": <int — count of distinct advice items in the feedback>,
        "emitted": <int — total directives across all roles>,
        "dropped": [
        {{ "advice_verbatim": "<copy from feedback>", "reason": "<one of: subjective_only | not_role_specific | unobservable_state | other>" }}
        ]
    }}
    }}

    Hard constraints:
    - "rationale_short" of every emitted directive MUST be a verbatim quote or close paraphrase of a sentence/phrase from the feedback above.
    - "id" must be unique within the file. Use ROLE-NNN format (SEER-001, SEER-002, GUARD-001, ...).
    - If a piece of advice applies to multiple roles, emit one directive per role.
    - "total_advice_pieces" must equal "emitted" + len("dropped").
    - There is NO per-role cap. Emit as many as the feedback supports.

    No extra text, no markdown, no code fences.
    """

        parsed = None
        for attempt in range(MAX_RETRIES):
            resp = self._call_model(system, prompt, max_tokens=4000)
            by_role = resp.get("by_role")
            coverage = resp.get("coverage")
            if (
                isinstance(by_role, dict)
                and all(isinstance(v, list) for v in by_role.values())
                and isinstance(coverage, dict)
            ):
                parsed = {"by_role": by_role, "coverage": coverage}
                break
            print(f"Warning: Coach failed to generate valid directives JSON (Attempt {attempt + 1}/{MAX_RETRIES})")

        if parsed is None:
            print("Warning: Coach failed to emit structured directives; skipping SDS for this game.")
            return

        # Merge with existing directives file if one is already on disk for this game.
        # This protects against re-runs producing a shorter list than was previously captured.
        merged = self._merge_directives_with_existing(parsed)

        # Sanity-check coverage: total should equal emitted + dropped count.
        cov = merged["coverage"]
        emitted_count = sum(len(v) for v in merged["by_role"].values() if isinstance(v, list))
        dropped_count = len(cov.get("dropped", []))
        if cov.get("total_advice_pieces") != emitted_count + dropped_count:
            cov["coverage_warning"] = (
                f"total_advice_pieces={cov.get('total_advice_pieces')} but "
                f"emitted+dropped={emitted_count + dropped_count}"
            )
        cov["emitted"] = emitted_count

        write_to_file(self._coach_directives_path, json.dumps(merged, indent=2))


    def _merge_directives_with_existing(self, new_parsed: dict) -> dict:
        """If a directives file already exists for this game, merge the new
        output into it. Existing directives are preserved; new ones are added
        by unique id. Coverage's `dropped` list is concatenated and de-duped
        by `advice_verbatim`. `total_advice_pieces` becomes the max of old
        and new (we trust whichever pass saw more advice).

        On a fresh game (no existing file) this is a no-op pass-through.
        """
        existing_path = self._coach_directives_path
        if not existing_path.exists():
            return new_parsed

        try:
            existing = json.loads(existing_path.read_text())
        except Exception:
            return new_parsed  # corrupted file — overwrite

        existing_by_role = existing.get("by_role", {}) if isinstance(existing, dict) else {}
        new_by_role = new_parsed.get("by_role", {})

        merged_by_role: dict[str, list] = {}
        all_roles = set(existing_by_role.keys()) | set(new_by_role.keys())

        for role in all_roles:
            old_list = existing_by_role.get(role, []) if isinstance(existing_by_role.get(role, []), list) else []
            new_list = new_by_role.get(role, []) if isinstance(new_by_role.get(role, []), list) else []
            seen_ids = set()
            combined = []
            for d in list(old_list) + list(new_list):
                if not isinstance(d, dict):
                    continue
                did = d.get("id")
                if not did or did in seen_ids:
                    continue
                seen_ids.add(did)
                combined.append(d)
            merged_by_role[role] = combined

        # Merge coverage
        existing_cov = existing.get("coverage", {}) if isinstance(existing, dict) else {}
        new_cov = new_parsed.get("coverage", {})

        seen_dropped = set()
        merged_dropped = []
        for entry in (existing_cov.get("dropped", []) or []) + (new_cov.get("dropped", []) or []):
            if not isinstance(entry, dict):
                continue
            key = entry.get("advice_verbatim", "")
            if key and key not in seen_dropped:
                seen_dropped.add(key)
                merged_dropped.append(entry)

        merged_total = max(
            int(existing_cov.get("total_advice_pieces", 0) or 0),
            int(new_cov.get("total_advice_pieces", 0) or 0),
        )

        return {
            "by_role": merged_by_role,
            "coverage": {
                "total_advice_pieces": merged_total,
                "emitted": sum(len(v) for v in merged_by_role.values()),
                "dropped": merged_dropped,
            },
        }

    def _evaluate_directives_uptake(self, game_record: str) -> None:
        """Evaluate whether the PREVIOUS game's directives were followed by the
        villager side in THIS game. Writes directives_uptake.json.

        Players never see this file. It exists purely so we can measure uptake of
        natural-language coaching without leaking structured cues into the player
        decision-time prompt pipeline.

        After the LLM returns evaluations, reconciles against the input directive
        list — any directive the LLM dropped from its response gets a synthesized
        NOT_EVALUATED placeholder so the file shape always matches the input set.
        """
        prev_path = self._previous_directives_path()
        if prev_path is None:
            return  # game 1, or previous emission failed — nothing to evaluate

        try:
            previous_directives = json.loads(prev_path.read_text())
        except Exception as e:
            print(f"Warning: failed to load previous directives from {prev_path}: {e}")
            return

        # New schema: {"by_role": {...}, "coverage": {...}}.
        # Tolerant of legacy flat schema for backward compatibility.
        if isinstance(previous_directives, dict) and "by_role" in previous_directives:
            by_role_input = previous_directives["by_role"]
        else:
            by_role_input = previous_directives  # legacy flat shape

        flat_directives = []
        for role, directives in by_role_input.items():
            if not isinstance(directives, list):
                continue
            for d in directives:
                if isinstance(d, dict):
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
            "events that are not in the game record. You MUST return one "
            "evaluation entry for EVERY directive you are given — do not skip any."
        )

        prompt = f"""
    === Previous Game's Directives ===
    {json.dumps(flat_directives, indent=2)}

    === This Game's Full Record ===
    {game_record}

    For EACH directive above, produce one evaluation entry. You MUST return one
    entry per input directive — the count of evaluations must equal the count of
    input directives. Do NOT omit any directive from your response.

    Return a JSON object with this exact shape:

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
        "actual_player_behavior": "<full natural-language description, or null if situation did not arise>",
        "verdict": "FOLLOWED" | "IGNORED" | "PARTIALLY_FOLLOWED" | "NOT_APPLICABLE",
        "verdict_notes": "<full natural-language justification>"
        }},
        ...
    ]
    }}

    Verdict rules:
    - situation_arose=false -> verdict="NOT_APPLICABLE", arose_at_round=null, arose_at_phase=null, actual_player_behavior=null.
    - situation_arose=true and action matched the directive -> "FOLLOWED".
    - situation_arose=true and action clearly contradicted the directive -> "IGNORED".
    - situation_arose=true and action was consistent with the directive's intent but deviated from its literal prescription -> "PARTIALLY_FOLLOWED".
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
            # Total LLM failure — synthesize a full NOT_EVALUATED file so the
            # downstream analysis still sees one row per directive rather than
            # silently missing the whole game.
            evaluations = []

        # ────────────────────────────────────────────────────────────────────
        # Reconciliation: ensure every input directive has an output entry.
        # If the LLM dropped any from its response (or the entire call failed),
        # synthesize a NOT_EVALUATED placeholder so the file shape matches the
        # input set exactly. This is the fix for `evaluations` being incomplete.
        # ────────────────────────────────────────────────────────────────────
        returned_ids = {
            ev.get("directive_id")
            for ev in evaluations
            if isinstance(ev, dict) and ev.get("directive_id")
        }
        for d in flat_directives:
            did = d.get("id")
            if not did or did in returned_ids:
                continue
            evaluations.append({
                "directive_id": did,
                "role": d.get("role", "Unknown"),
                "directive_phase": d.get("phase", ""),
                "directive_trigger": d.get("trigger", ""),
                "directive_action": d.get("action", ""),
                "directive_rationale": d.get("rationale_short", ""),
                "situation_arose": None,
                "arose_at_round": None,
                "arose_at_phase": None,
                "actual_player_behavior": None,
                "verdict": "NOT_EVALUATED",
                "verdict_notes": (
                    "Evaluator LLM did not return a verdict for this directive. "
                    "Placeholder inserted by reconciliation."
                ),
            })

        if not evaluations:
            # No input directives AND no LLM output — nothing to write.
            return

        # Group by role for the output file
        by_role: dict[str, list] = {}
        for ev in evaluations:
            if not isinstance(ev, dict):
                continue
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