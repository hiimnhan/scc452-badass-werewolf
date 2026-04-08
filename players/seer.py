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
    """

    def __init__(
        self,
        name: str,
        model: BaseChatModel,
        role: Role = Role.SEER,
        system_prompt: str = SEER_PROMPT_TEMPLATE,
        personality: str = "",          # expansion hook — no-op until archetypes configured
    ) -> None:
        super().__init__(
            name=name,
            role=role,
            model=model,
            system_prompt=system_prompt,
            personality=personality,
        )
        self._investigations: list[dict] = []
        self._role_revealed: bool = False

    # Public querry helpers

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

    def _confirmed_label(self, player_name: str) -> str:
        """Short label for prompt context."""
        result = self._is_confirmed(player_name)
        if result is True:
            return f"{player_name} [CONFIRMED WOLF — investigated]"
        if result is False:
            return f"{player_name} [CONFIRMED INNOCENT — investigated]"
        return player_name

    def _format_investigation_results(self) -> str:
        """
        Neutral listing of all investigation results as private facts.
        No strategic directives — the LLM decides how and when to use this.
        """
        if not self._investigations:
            return "None yet — you have not investigated anyone this game."
        lines = []
        for inv in self._investigations:
            result = "is a werewolf" if inv["is_wolf"] else "is NOT a werewolf"
            lines.append(f"  {inv['player']}: {result}")
        return "\n".join(lines)

    # SPECIAL NIGHT ACTIONS

    def unmask(self, alive_players: List[str]) -> tuple[str, dict]:
        """
        Night action: choose one alive player to secretly investigate.
        Returns (target_name, log_dict) for the moderator.
        The moderator then calls reveal_and_update(target_name, is_wolf).

        Guards:
            — dead seer cannot act
            — None or empty alive_players returns error (no crash)
            — all alive players already investigated: falls back to
            full available pool (soft preference, not hard block)
        """
        # dead seer must not act
        if not self._is_alive:
            return "", {"error": "Seer is dead and cannot investigate."}

        # guard against None or empty list
        if not alive_players:
            return "", {"error": "No alive players provided."}

        # Seer cannot investigate themselves
        available_targets = [p for p in alive_players if p != self._name]
        if not available_targets:
            return "", {"error": "No valid targets to investigate (only self is alive)."}

        # Soft preference: show uninvestigated players as target_pool first.
        # If all are investigated, fall back to full available pool.
        # The LM may still pick any player in available_targets — validated below.
        already_investigated = {inv["player"] for inv in self._investigations}
        uninvestigated = [
            p for p in available_targets if p not in already_investigated]
        target_pool = uninvestigated if uninvestigated else available_targets

        # Format investigation history for the prompt
        if self._investigations:
            investigated_notes = "; ".join(
                f"{inv['player']} ({'wolf' if inv['is_wolf'] else 'not wolf'})"
                for inv in self._investigations
            )
        else:
            investigated_notes = "None yet."

        prompt = SEER_UNMASK_PROMPT_TEMPLATE.format(
            name=self._name,
            target_pool=", ".join(target_pool),
            investigated_notes=investigated_notes,
            suspicions=(
                ", ".join(
                    f"{self._confirmed_label(p)} ({s:.2f})"
                    for p, s in self._suspicions.items()
                )
                if self._suspicions
                else "No suspicions recorded yet."
            ),
            self_reflections=self._self_reflections or "No specific guidelines yet.",
        )

        resp = self.call_model(prompt, max_tokens=300)
        target = resp.get("target", "")

        # Validation: must be in available_targets (preference for uninvestigated is soft)
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

        self._add_note(
            f"Night investigation: chose to investigate {target}. "
            f"Reason: {resp.get('analysis', 'No analysis provided.')}"
        )
        return target, resp

    def reveal_and_update(self, player_name: str, is_wolf: bool) -> None:
        """
        Called by the moderator after checking the investigated player's role.

        Args:
            player_name: player that was investigated (non-empty string)
            is_wolf:     True  → Werewolf
                         False → NOT a Werewolf (exact role remains unknown)

        The seer learns ONLY the wolf/not-wolf binary.
        The suspicion score is set definitively and protected from overwrite
        by the detect_deception() override below.

        Guards:
            Case 8 — None/empty player_name raises ValueError immediately
            Case 1 — duplicate call with same result is idempotent (no side effects)
                     duplicate call with conflicting result raises ValueError
        """
        # Case 8: guard invalid player_name
        if not player_name:
            raise ValueError(
                "reveal_and_update: player_name must be a non-empty string."
            )

        # Case 1: idempotency guard
        existing = self._is_confirmed(player_name)
        if existing is not None:
            if existing != is_wolf:
                raise ValueError(
                    f"reveal_and_update: conflicting result for '{player_name}'. "
                    f"Previously recorded is_wolf={existing}, "
                    f"now receiving is_wolf={is_wolf}. "
                    "This is a game logic error."
                )
            # Same result called again — silently idempotent, no side effects.
            return

        result_text = "a werewolf" if is_wolf else "NOT a werewolf"

        # Append-only: never mutated after this point
        self._investigations.append(
            {"player": player_name, "is_wolf": is_wolf})

        # Definitive suspicion score — protected from LLM inference by override below
        self._suspicions[player_name] = 1.0 if is_wolf else 0.0

        self._add_note(
            f"Seer result [CONFIRMED]: {player_name} is {result_text}."
        )

    # OVERRIDDEN INHERITED ACTIONS

    def detect_deception(
        self,
        speaker_name: str,
        statement: str,
        context: str = "",
        history: List[dict] = None,
    ) -> dict:
        """
        Override to protect confirmed investigation scores from LM inference.

        If this speaker has already been investigated, their suspicion score is
        FIXED ground truth from the moderator. The LM analysis still runs for
        chain-of-thought and note value, but its suspicion_level output is
        discarded and the confirmed score is restored immediately after.
        """
        if history is None:
            history = []

        confirmed = self._is_confirmed(speaker_name)
        if confirmed is not None:
            # Run analysis for chain-of-thought / game log value
            resp = super().detect_deception(speaker_name, statement, context, history)
            fixed_score = 1.0 if confirmed else 0.0
            original_llm_suspicion = resp.get(
                "suspicion_level")  # save before overwrite
            # Restore — super() just overwrote the ground-truth value
            self._suspicions[speaker_name] = fixed_score
            resp["suspicion_level"] = fixed_score
            resp["_confirmed_by_investigation"] = True
            resp["_original_llm_suspicion"] = original_llm_suspicion
            return resp

        # Not yet investigated — normal inference applies, score updates freely
        return super().detect_deception(speaker_name, statement, context, history)

    def vote(self, alive_players: List[str]) -> tuple[str, dict]:
        """
        Override to provide investigation results as private context in the prompt.

        The LM decides freely how to vote — including voting for someone other
        than a confirmed wolf to hide the seer's identity. No forced vote.
        """
        available_targets = [p for p in alive_players if p != self._name]
        if not available_targets:
            return "", {"error": "No available targets to vote for."}

        prompt = SEER_VOTE_PROMPT_TEMPLATE.format(
            name=self._name,
            available_targets=", ".join(available_targets),
            investigation_results=self._format_investigation_results(),
            suspicions=(
                ", ".join(f"{p} ({s:.2f})" for p,
                          s in self._suspicions.items())
                if self._suspicions
                else "None recorded yet."
            ),
            self_reflections=self._self_reflections or "No specific guidelines yet.",
            notes=", ".join(self._current_game_notes) or "No notes yet.",
        )

        resp = self.call_model(prompt)
        target = resp.get("vote", "")

        if target not in available_targets:
            if "raw" in resp:
                for t in available_targets:
                    if t in resp["raw"]:
                        target = t
                        break
            if target not in available_targets:
                target = available_targets[0]
                resp["vote"] = target
                resp["fallback"] = "Used first available target due to invalid response."

        self._add_note(
            f"Voted to eliminate {target}. "
            f"Reason: {resp.get('analysis', 'No analysis provided.')}"
        )
        return target, resp

    def debate(self, dialogue_history: List[list]) -> tuple[str, dict]:
        """
        Override to:
        - Provide investigation results as neutral private context
        - Track whether seer has already publicly revealed their role
          (state tracking only — not forcing the LM to act on it)

        The LM decides freely when and whether to mention investigation results.
        """
        history_text = "\n".join(f"{s}: {t}" for s, t in dialogue_history)

        prompt = SEER_DEBATE_PROMPT_TEMPLATE.format(
            name=self._name,
            dialogue_history=history_text or "(no previous dialogue)",
            investigation_results=self._format_investigation_results(),
            role_revealed=self._role_revealed,
            suspicions=(
                "\n".join(f"  {p}: {s:.2f}" for p,
                          s in self._suspicions.items())
                if self._suspicions
                else "  None recorded yet."
            ),
            self_reflections=self._self_reflections or "No specific guidelines yet.",
            notes=", ".join(self._current_game_notes) or "No notes yet.",
        )

        resp = self.call_model(prompt, max_tokens=400)
        statement = resp.get("statement", "").strip()

        if not statement:
            raw = resp.get("raw", "")
            match = re.search(r'"statement"\s*:\s*"([^"]+)"', raw)
            statement = match.group(1).strip() if match else ""
            if not statement:
                return "", {"error": "No valid statement generated."}

        # State tracking: detect if this statement reveals the seer role.
        # This prevents self-contradiction in later turns by
        # informing the prompt that revelation has occurred.
        role_keywords = ["i am the seer",
                         "as the seer", "i'm the seer", "seer here"]
        if not self._role_revealed and any(
            kw in statement.lower() for kw in role_keywords
        ):
            self._role_revealed = True
            self._add_note("Revealed Seer role publicly during debate.")

        self._add_note(f"Contributed to debate: {statement}")
        return statement, resp
