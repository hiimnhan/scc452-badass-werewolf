import re
from typing import List, Optional

from langchain_core.language_models import BaseChatModel

from players.base_player import BasePlayer, Role
from prompt import (
    SEER_DEBATE_PROMPT_TEMPLATE,
    SEER_PROMPT_TEMPLATE,
    SEER_UNMASK_PROMPT_TEMPLATE,
    SEER_VOTE_PROMPT_TEMPLATE,
)


class Seer(BasePlayer):
    """
    The Seer investigates one player per night.
    The moderator returns ONLY True/False (is_wolf) — never the exact role.

    Extra state beyond BasePlayer:
        _investigations  list[dict]  ground-truth records {"player", "is_wolf"}
                                     append-only after reveal_and_update()
        _role_revealed   bool        True once the seer publicly claims their
                                     role in debate; prevents contradictory
                                     re-revelation in later turns

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
        role: Role = Role.SEER,
        system_prompt: str = SEER_PROMPT_TEMPLATE,
        personality: str = "",
    ) -> None:
        super().__init__(
            name=name,
            role=role,
            model=model,
            game_id=game_id,
            system_prompt=system_prompt,
            personality=personality,
        )
        self._investigations: list[dict] = []
        self._role_revealed: bool = False

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

    def unmask(self, alive_players: List[str], round_num: int) -> tuple[str, dict]:
        """Night action: choose one alive player to secretly investigate.

        Returns (target_name, log_dict) for the moderator.
        The moderator then calls reveal_and_update(target_name, is_wolf, round_num).

        Guards:
            — dead seer cannot act
            — None or empty alive_players returns error (no crash)
            — all alive players already investigated: falls back to full available pool
        """
        if not self._is_alive:
            return "", {"error": "Seer is dead and cannot investigate."}

        if not alive_players:
            return "", {"error": "No alive players provided."}

        # Seer cannot investigate themselves
        available_targets = [p for p in alive_players if p != self._name]
        if not available_targets:
            return "", {"error": "No valid targets to investigate (only self is alive)."}

        # Soft preference: show uninvestigated first. LLM may still pick any available target.
        already_investigated = {inv["player"] for inv in self._investigations}
        uninvestigated = [
            p for p in available_targets if p not in already_investigated]
        target_pool = uninvestigated if uninvestigated else available_targets

        prompt = SEER_UNMASK_PROMPT_TEMPLATE.format(
            name=self._name,
            target_pool=", ".join(target_pool),
            investigation_results=self._format_investigation_results(),
            note=self._note,
        )

        resp = self.call_model(prompt, max_tokens=300)
        target = resp.get("target", "")

        # Validation: must be in available_targets (uninvestigated preference is soft)
        if target not in available_targets:
            if "raw" in resp:
                for p in available_targets:
                    if p in resp["raw"]:
                        target = p
                        break
            if target not in available_targets:
                target = target_pool[0]
                resp["target"] = target
                resp["fallback"] = "Used first available target due to invalid response."

        self._record_own_action(
            round_num,
            "Night",
            f"Chose to investigate {target}. "
            f"Reason: {resp.get('analysis', 'No analysis provided.')}",
        )
        return target, resp

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
            raise ValueError(
                "reveal_and_update: player_name must be a non-empty string."
            )

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
        self._investigations.append(
            {"player": player_name, "is_wolf": is_wolf})

        # Lock the suspicion entry — score and reason are ground truth.
        # Protected from LLM overwrite by _update_suspicion() override below.
        self._suspicion[player_name] = {
            "score":  1.0 if is_wolf else 0.0,
            "reason": self.CONFIRMED_WOLF_REASON if is_wolf else self.CONFIRMED_INNOCENT_REASON,
        }

        result_text = "a werewolf" if is_wolf else "NOT a werewolf"
        self._record_own_action(
            round_num,
            "Night",
            f"Investigation result [CONFIRMED]: {player_name} is {result_text}.",
        )

    # Overridden inherited actions

    def _update_suspicion(self, speaker_name: str, statement: str) -> dict:
        """Override to protect confirmed investigation results from LLM inference.

        Players confirmed by investigation have their score AND reason locked.
        The base implementation still runs (to update every other player
        normally and to produce the chain-of-thought), then confirmed entries
        are restored immediately after.
        """
        # Snapshot confirmed entries before base overwrites them
        locked: dict = {}
        for name in self._suspicion:
            if self._is_confirmed(name) is not None:
                locked[name] = dict(self._suspicion[name])

        # Base updates every player freely
        resp = super()._update_suspicion(speaker_name, statement)

        # Restore confirmed entries — moderator's answer is ground truth
        for name, entry in locked.items():
            self._suspicion[name] = entry

        if locked:
            resp["_confirmed_preserved"] = list(locked.keys())
        return resp

    def vote(self, alive_players: List[str]) -> tuple[str, dict]:
        """Override to include investigation results as private context.

        The LLM decides freely how to vote — including voting for someone
        other than a confirmed wolf to hide the seer's identity.
        """
        available = [p for p in alive_players if p != self._name]
        if not available:
            return "", {"error": "No available targets to vote for."}

        prompt = SEER_VOTE_PROMPT_TEMPLATE.format(
            name=self._name,
            available_targets=", ".join(available),
            investigation_results=self._format_investigation_results(),
            note=self._note,
        )

        resp = self.call_model(prompt, max_tokens=200)
        target = resp.get("vote", "")

        if target not in available:
            if "raw" in resp:
                for t in available:
                    if t in resp["raw"]:
                        target = t
                        break
            if target not in available:
                return "", {
                    "error": f"Invalid vote '{target}'. Must be one of: {available}"
                }

        return target, resp

    def debate(self) -> tuple[str, dict]:
        """Override to include investigation results and track _role_revealed.

        Signature matches BasePlayer.debate() — no dialogue_history parameter;
        per-statement context is already in self._note via _update_suspicion.

        The LLM decides freely when and whether to reveal.
        """
        prompt = SEER_DEBATE_PROMPT_TEMPLATE.format(
            name=self._name,
            investigation_results=self._format_investigation_results(),
            role_revealed=self._role_revealed,
            note=self._note,
        )

        resp = self.call_model(prompt, max_tokens=200)
        statement = resp.get("statement", "").strip()

        if not statement:
            raw = resp.get("raw", "")
            match = re.search(r'"statement"\s*:\s*"([^"]+)"', raw)
            statement = match.group(1).strip() if match else ""
            if not statement:
                return "", {"error": "No valid statement generated."}

        # State tracking: detect public role revelation so later turns
        # see role_revealed=True in the prompt and avoid contradicting.
        role_keywords = ["i am the seer",
                         "as the seer", "i'm the seer", "seer here"]
        if not self._role_revealed and any(kw in statement.lower() for kw in role_keywords):
            self._role_revealed = True

        return statement, resp
