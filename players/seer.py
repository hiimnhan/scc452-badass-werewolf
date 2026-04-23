from typing import List, Optional
from langchain_core.language_models import BaseChatModel
from players.base_player import BasePlayer, Role
from prompt import SEER_PROMPT_TEMPLATE, SEER_UNMASK_PROMPT_TEMPLATE
from constants import MAX_RETRIES


class Seer(BasePlayer):
    """
    The Seer investigates one player per night.
    The moderator returns ONLY True/False (is_wolf) — never the exact role.

    Extra state beyond BasePlayer:
        _investigations  list[dict]  ground-truth records {"player", "is_wolf"}
                                     append-only after reveal_and_update()

    Confirmed investigation results are stored in self._suspicion[name] with
    score 1.0 or 0.0 and a fixed reason string. The _update_suspicion override
    protects these entries from being overwritten by LLM inference on later
    day-phase statements.
    """

    CONFIRMED_WOLF_REASON = "CONFIRMED wolf via seer investigation."
    CONFIRMED_INNOCENT_REASON = "CONFIRMED not a wolf via seer investigation."

    def __init__(
        self,
        name: str,
        model: BaseChatModel,
        game_id: str = "",
        scenario: str = "baseline",
        role: Role = Role.SEER,
        system_prompt: str = SEER_PROMPT_TEMPLATE,
        personality: str = "",
    ) -> None:
        super().__init__(
            name=name,
            role=role,
            model=model,
            game_id=game_id,
            scenario=scenario,
            system_prompt=system_prompt,
            personality=personality,
        )
        self._investigations: list[dict] = []

    # Public query helpers

    def get_confirmed_wolves(self) -> list[str]:
        """Names of players confirmed as wolves by investigation."""
        return [inv["player"] for inv in self._investigations if inv["is_wolf"]]

    def get_confirmed_innocents(self) -> list[str]:
        """Names of players confirmed as NOT wolves by investigation."""
        return [inv["player"] for inv in self._investigations if not inv["is_wolf"]]

    def _is_confirmed(self, player_name: str) -> Optional[bool]:
        """
        Return the ground-truth result for a player, or None if not investigated.
        None  → not yet investigated
        True  → confirmed wolf
        False → confirmed innocent (exact role still unknown)
        """
        record = next(
            (inv for inv in self._investigations if inv["player"] == player_name),
            None,
        )
        return record["is_wolf"] if record is not None else None

    def _format_investigation_results(self) -> str:
        """Neutral listing of all investigation results as private facts.
        No strategic directives — the LLM decides how and when to use this.
        """
        if not self._investigations:
            return "None yet — you have not investigated anyone this game."
        lines = []
        for inv in self._investigations:
            result = "is a werewolf" if inv["is_wolf"] else "is NOT a werewolf"
            lines.append(f"  {inv['player']}: {result}")
        return "\n".join(lines)

    # Special night actions

    def unmask(self, alive_players: list[str], round_num: int) -> tuple[str, str, dict]:
        if not self._is_alive:
            return "None", "Seer is dead.", {"error": "Seer is dead and cannot investigate."}

        if not alive_players:
            return "None", "No alive players provided.", {"error": "No alive players provided."}

        # Seer cannot investigate themselves
        available_targets = [p for p in alive_players if p != self._name]
        if not available_targets:
            return "None", "No valid targets.", {"error": "No valid targets to investigate (only self is alive)."}

        # Soft preference: show uninvestigated first. LLM may still pick any available target.
        already_investigated = {inv["player"] for inv in self._investigations}
        uninvestigated = [p for p in available_targets if p not in already_investigated]
        target_pool = uninvestigated if uninvestigated else available_targets

        prompt = SEER_UNMASK_PROMPT_TEMPLATE.format(
            name=self._name,
            target_pool=", ".join(target_pool),
            investigation_results=self._format_investigation_results(),
            note=self._note,
        )

        target = "None"
        analysis = "Default analysis: LLM failed to provide reasoning."

        for attempt in range(MAX_RETRIES):
            resp = self.call_model(prompt, max_tokens=300)
            
            if "target" in resp and "analysis" in resp:
                target = resp["target"]
                analysis = resp["analysis"]
                break
            else:
                print(f"Warning: {self._name} ({self._role.value}) failed to generate valid target JSON (Attempt {attempt + 1}/{MAX_RETRIES})")

        # Validation: must be in available_targets
        if target not in available_targets:
            # 1st Fallback: Raw text extraction
            if "raw" in resp:
                for p in available_targets:
                    if p in resp["raw"]:
                        target = p
                        analysis = "Extracted target from raw text due to JSON failure."
                        resp["target"] = target
                        resp["analysis"] = analysis
                        break
            
            # 2nd Fallback: Random selection
            if target not in available_targets:
                import random
                target = random.choice(target_pool)
                analysis = "Forced random target due to invalid LLM response."
                resp["target"] = target
                resp["analysis"] = analysis
                resp["fallback"] = analysis

        self.record_own_action(
            round_num,
            "Night",
            f"Chose to investigate {target}. Reason: {analysis}",
        )
        return target, analysis, resp

    def reveal_and_update(self, player_name: str, is_wolf: bool, round_num: int) -> None:
        """Moderator calls this after checking the investigated player's role.

        Args:
            player_name: player that was investigated (non-empty string)
            is_wolf:     True  → Werewolf
                         False → NOT a Werewolf (exact role unknown)
            round_num:   current round number for the own-action log

        The seer learns ONLY the wolf/not-wolf binary.
        The suspicion score and reason are locked and protected from LLM
        overwrite by the _update_suspicion override below.

        Guards:
            — None/empty player_name raises ValueError immediately
            — duplicate call with same result is idempotent (no side effects)
            — duplicate call with conflicting result raises ValueError
        """
        if not player_name:
            raise ValueError("reveal_and_update: player_name must be a non-empty string.")

        existing = self._is_confirmed(player_name)
        if existing is not None:
            if existing != is_wolf:
                raise ValueError(
                    f"reveal_and_update: conflicting result for '{player_name}'. "
                    f"Previously recorded is_wolf={existing}, "
                    f"now receiving is_wolf={is_wolf}. "
                    "This is a game logic error."
                )
            # Same result called again — silently idempotent.
            return

        # Append-only: never mutated after this point
        self._investigations.append({"player": player_name, "is_wolf": is_wolf})

        # Lock the suspicion entry — score and reason are ground truth.
        # Protected from LLM overwrite by _update_suspicion() override below.
        self._suspicion[player_name] = {
            "score": 1.0 if is_wolf else 0.0,
            "reason": self.CONFIRMED_WOLF_REASON if is_wolf else self.CONFIRMED_INNOCENT_REASON,
        }

        result_text = "a werewolf" if is_wolf else "NOT a werewolf"
        self.record_own_action(
            round_num,
            "Night",
            f"Investigation result [CONFIRMED]: {player_name} is {result_text}.",
        )
