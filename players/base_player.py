from enum import Enum
import json
from typing import List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from abc import ABC
from pathlib import Path
from constants import COACH_FEEDBACK_FILENAME
from config import SCENARIO_CONFIG
from utils import write_to_file


# ============================================
# Game Rules
# Injected once into every player's system prompt at init.
# Never re-injected mid-game — it never changes.
# ============================================

GAME_RULES = """
=== WEREWOLF — OFFICIAL GAME RULES ===
 
FACTIONS
  Villagers win if all Werewolves are eliminated.
  Werewolves win if they equal or outnumber Villagers.
 
PHASES (repeated each round until a faction wins)
  1. NIGHT — roles act in this order:
       a. Werewolf Debate    — wolves privately agree on a target.
       b. Werewolf Eliminate — wolves kill their chosen target.
       c. Guard Protect      — Guard secretly shields one player.
       d. Seer Investigate   — Seer learns if one player is a Werewolf.
       e. Witch Act          — Witch may Save the wolf target and/or Poison any player.
       f. Resolve Night      — apply kills, saves, and poisons; announce the outcome.
  2. DAY — all alive players participate:
       a. Debate — players discuss and accuse (bidding determines speaker order).
       b. Vote   — each player votes to exile one suspect (the vote is only executed if the number of votes for one person is greater or equal to half of the total alive players. Ties meeting this threshold are broken randomly).
       c. Exile  — the player with the most votes is eliminated.
 
ROLES
  Villager — no night action; reasons from public information only.
  Werewolf — knows other wolves; eliminates one non-wolf per night.
  Seer     — learns one player's alignment (wolf / not wolf) each night.
  Guard    — protects one player per night; cannot protect the same player twice in a row.
  Witch    — has one Save potion (cancels wolf kill) and one Poison potion (kills any player).
             Each potion is single-use.
             The Poison potion bypasses the Guard — a poisoned player dies even if protected.
             Note: The Witch does not know who the Guard protected. She may accidentally waste her Save potion on a player already protected by the Guard.
 
INFORMATION RULES
  • The eliminated player's role is NOT revealed publicly.
  • The Guard's protection target is never announced.
  • The Seer's results are private until the Seer chooses to reveal them.
  • The Witch sees who the wolves targeted ONLY if her Save potion is still available.
    Once the Save potion has been used, the Witch no longer learns the wolf target.
=== END OF RULES ===
"""


# ============================================
# Roles
# ============================================

class Role(Enum):
    VILLAGER = "Villager"
    WEREWOLF = "Werewolf"
    SEER     = "Seer"
    GUARD    = "Guard"
    WITCH    = "Witch"


VILLAGER_SIDE = {Role.VILLAGER, Role.SEER, Role.GUARD, Role.WITCH}
WOLF_SIDE     = {Role.WEREWOLF}


# ============================================
# Base Player
# ============================================

class BasePlayer(ABC):
    """
    Abstract base for all Werewolf player agents.

    ── Memory model ────────────────────────────────────────────────────────
    Every call_model() sends two messages to the LLM:

      SystemMessage  →  self._setup_prompt
                        Built ONCE at __init__ from stable sources:
                          • role definition prompt
                          • personality
                          • strategy loaded from [name]_strategy.txt
                          • GAME_RULES
                        Re-sent on every call — the LLM has no memory between calls. The Python object IS the memory.

      HumanMessage   →  self._note   (assembled fresh per action)
                        Built each time from in-memory state:
                          • _game_summary_entries  (public events + own actions)
                          • _suspicions            (score + reason per player)

    ── File layout ─────────────────────────────────────────────────────────
      strategies/  {name}_strategy.txt                              persists across games
      game_logs/ game_{game_id}/   {name}_game_{game_id}_note.txt   per-game compiled note
      game_logs/ game_{game_id}/   coach_feedback.txt               written by Coach, read here

    ── Lifecycle ───────────────────────────────────────────────────────────
      Game start  : init_suspicions(other_players)
      Each night  : receive_announcement() from game engine after resolve
      Each debate : _update_suspicion() after each statement
      Round end   : _compile_note()  — only disk flush during a game
      Game end    : _update_strategy(self_analyze, coaching)
      Next game   : reset_game_state(game_id, other_players)
    """

    def __init__(
        self,
        name: str,
        role: Role,
        model: BaseChatModel,
        game_id: str = "",
        scenario: str = "baseline",
        is_alive: bool = True,
        system_prompt: str = "", # role definition template; receives {name}
        personality: str = "",
    ) -> None:
        self._name = name
        self._role = role
        self._model = model
        self._game_id = game_id
        self._scenario = scenario
        self._self_analyze = SCENARIO_CONFIG[self._scenario]['self_analyze']
        self._coaching = SCENARIO_CONFIG[self._scenario]['coaching']
        self._is_alive = is_alive
        self._role_definition = system_prompt
        self._personality = personality

        # Intra-game state — reset by reset_game_state()
        self._game_summary_entries: List[str] = []
        self._suspicion: dict = {} # {name: {"score": float, "reason": str}}

        # System prompt — built once, sent on every call
        strategy = self._load_strategy()
        self._setup_prompt = self._build_setup_prompt(strategy)

    # ── Dunder ──────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name}, role={self._role.value}, alive={self._is_alive})"

    # ── File path helpers ───────────────────────────────────────────────

    def _base_dir(self) -> Path:
        return Path(__file__).parent.parent

    @property
    def _strategy_path(self) -> Path:
        return (self._base_dir() / "strategies" / self._scenario / f"{self._name}_strategy.txt").resolve()

    @property
    def _note_path(self) -> Path:
        return (self._base_dir() / "game_logs" / self._scenario / f"game_{self._game_id}" / f"{self._name}_{self._role.value}_note.txt").resolve()

    @property
    def _feedback_path(self) -> Path:
        return (self._base_dir() / "game_logs" / self._scenario / f"game_{self._game_id}" / COACH_FEEDBACK_FILENAME).resolve()

    # ── Setup helpers ───────────────────────────────────────────────────

    def _build_setup_prompt(self, strategy: str) -> str:
        """Assemble the system message from all stable identity components.
        Called once at __init__; refreshed by _write_strategy() after a game.
        """
        parts = [
            self._role_definition.format(name=self._name) if self._role_definition else "",
            f"Your personality: {self._personality}" if self._personality else "",
            f"Your strategy from previous games:\n{strategy}" if strategy else "",
            GAME_RULES,
        ]
        return "\n\n".join(p.strip() for p in parts if p.strip())

    def _load_strategy(self) -> str:
        """Read [name]_strategy.txt from disk.
        Returns empty string on first game (file does not exist yet).
        """
        path = self._strategy_path
        if not path.exists():
            return ""
        try:
            return path.read_text().strip()
        except Exception:
            return ""

    # ── Intra-game: initialisation ──────────────────────────────────────

    def init_suspicions(self, other_players: List[str]) -> None:
        """Seed suspicion dict for all other players at game start.
        Everyone begins at 0.5 (completely unknown) with no reason.
        Call once from game setup before Round 1.
        """
        self._suspicion = {
            p: {"score": 0.5, "reason": ""}
            for p in other_players
            if p != self._name
        }

    # ── Intra-game: game summary ────────────────────────────────────────

    def receive_announcement(self, round_num: int, phase: str, announcement: str) -> None:
        """Receive a public game-engine announcement and store it.

        Called by game nodes (resolve_night_node, exile_node) for ALL alive players. Role subclasses may override to personalise the wording
        (e.g. Witch appending "— poisoned by me").

        Args:
            round_num:    Current round (1-indexed).
            phase:        "Night" | "Day".
            announcement: Factual string, e.g. "Dave was eliminated."
        """
        self._game_summary_entries.append(f"Round {round_num} {phase}: {announcement}")

    def record_own_action(self, round_num: int, phase: str, event: str) -> None:
        """Append a private action to the game summary.
        Used by role action methods for events only this player knows
        (e.g. Witch recording her own potion use, Seer logging an investigation).
        NOT for public announcements — use receive_announcement() for those.
        """
        self._game_summary_entries.append(f"Round {round_num} {phase}: {event}")

    @property
    def _game_summary(self) -> str:
        """Formatted game summary string for prompt injection."""
        if not self._game_summary_entries:
            return "No events recorded yet."
        return "\n".join(f"- {e}" for e in self._game_summary_entries)

    # ── Intra-game: suspicion ───────────────────────────────────────────

    def update_suspicion_from_statement(self, speaker_name: str, statement: str) -> dict:
        """Update suspicion scores for ALL players after a statement is made.

        The LLM receives the speaker's statement AND the existing note, and is asked to 
        EXTEND the reasons for any relevant players. 
        In-memory only; call _compile_note() at round end to flush to disk.

        Returns the raw LLM response dict.
        """
        prompt = f"""
You are {self._name} ({self._role.value}).
{speaker_name} just said: "{statement}"

{self._note}

Analyse this new statement and how it impacts your read on EVERY player. Consider:
1. Does it contradict prior behaviour or claims?
2. Does it link {speaker_name} to anyone else (e.g., defending or accusing them)?
3. Does it help or hurt the villager side?

Extend the reason field for the players — do not erase prior notes.

Respond with ONLY a JSON object using this exact structure:
{{
  "chain_of_thought": "your private reasoning about how this statement connects players (<=40 words)",
  "updates": {{
    "{speaker_name}": {{"score": 0.0 to 1.0, "reason": "cumulative behavioural notes (<=40 words)"}},
    "AnotherPlayer": {{"score": 0.0 to 1.0, "reason": "updated notes if affected, or previous notes (<=40 words)"}}
  }}
}}
Include ALL other players you are tracking in the "updates" dictionary.
No extra text, no markdown, no code fences.
"""

        resp = self.call_model(prompt, max_tokens=800)

        # Extract the updates dictionary from the AI's response
        updates = resp.get("updates", {})
        
        # Loop through every player the AI returned
        for player, data in updates.items():
            # Only update if they are actually in our active tracking dictionary
            if player in self._suspicion:
                current = self._suspicion[player]
                
                # Safely extract the new score and reason, falling back to old ones if missing
                try:
                    new_score = max(0.0, min(1.0, float(data.get("score", current["score"]))))
                except (ValueError, TypeError):
                    new_score = current["score"]
                    
                new_reason = data.get("reason", current["reason"])

                # Apply the update
                self._suspicion[player] = {"score": new_score, "reason": new_reason}

        return resp
    
    def update_suspicion_from_vote(self, current_round_vote_logs: list[str], exiled_player: str, game_status: str) -> dict:
        """Analyze the results of the voting phase and update suspicion scores for all players."""
        
        # If no one voted, skip the analysis
        if not current_round_vote_logs:
            return {"error": "No votes to analyze"}

        # Format the voting results into a clean string
        voting_summary = "\n".join(current_round_vote_logs)
        voting_summary += f"\n\nVote outcome: {exiled_player} is exiled." if exiled_player else "\n\nVote outcome: No one is exiled."
        voting_summary += f"\n{game_status}"
        
        prompt = f"""
You are {self._name} ({self._role.value}).
The daily vote just concluded. Here is how everyone voted:

{voting_summary}

Here is your current knowledge:
{self._note}

Analyse these voting patterns and how they impact your suspicion scores for EVERY player. Consider:
1. Did Werewolves coordinate their votes (bandwagoning) on a single target?
2. Did anyone vote defensively to save themselves?
3. Does someone's vote contradict their previous statements or accusations?

Provide an updated score and a concise reason justifying your read on them based on this voting data.

Respond with ONLY a JSON object using this exact structure:
{{
  "chain_of_thought": "your private reasoning about the voting patterns (<=40 words)",
  "updates": {{
    "PlayerA": {{"score": 0.0 to 1.0, "reason": "justification based on who they voted for (<=40 words)"}},
    "PlayerB": {{"score": 0.0 to 1.0, "reason": "updated reason if affected (<=40 words)"}}
  }}
}}
Include ALL other players you are tracking in the "updates" dictionary.
No extra text, no markdown, no code fences.
"""
        
        # Use 800 max_tokens since it outputs updates for all players
        resp = self.call_model(prompt, max_tokens=800)

        # Extract and apply the updates (using the plural _suspicions we fixed earlier!)
        updates: dict = resp.get("updates", {})
        
        for player, data in updates.items():
            if player in self._suspicion:
                current = self._suspicion[player]
                
                try:
                    new_score = max(0.0, min(1.0, float(data.get("score", current["score"]))))
                except (ValueError, TypeError):
                    new_score = current["score"]
                    
                new_reason = data.get("reason", current["reason"])
                
                self._suspicion[player] = {"score": new_score, "reason": new_reason}

        return resp

    # ── Intra-game: note (the human-message context block) ──────────────

    def _format_suspicion_block(self) -> str:
        """Format self._suspicion dict into a readable string for prompt injection.
        Sorted highest score first so the most suspicious players are prominent.
        """
        if not self._suspicion:
            return "No suspicion data yet."
        lines = []
        for name, data in sorted(
            self._suspicion.items(),
            key=lambda x: x[1]["score"],
            reverse=True,
        ):
            reason = f' — {data["reason"]}' if data["reason"] else ""
            lines.append(f'  {name}: {data["score"]:.2f}{reason}')
        return "\n".join(lines)

    @property
    def _note(self) -> str:
        """The player's full working note for prompt injection.
        Used as the human-message context in vote(), debate(), _get_bid().
        Combines game summary and suspicion scores.
        """
        return (
            "=== Game Summary (your observations so far) ===\n"
            f"{self._game_summary}\n\n"
            "=== Suspicion Scores (0.0 = innocent → 1.0 = wolf) ===\n"
            f"{self._format_suspicion_block()}"
        )

    def compile_note(self) -> None:
        """Flush the current note to [name]_game_{id}_note.txt.
        Call at round end (after resolve_night_node and after exile_node).
        This is the ONLY disk write during a game.
        """
        write_to_file(self._note_path, self._note)

    # ── Intra-game: decision actions ────────────────────────────────────

    def get_bid(self) -> tuple[int, dict]:
        """Return a bid score 0-10: how urgently this player wants to speak next.

        High suspicion of a specific player → high urgency.
        Low variance / no clear target → low urgency.

        Returns (bid_score: int, log_dict: dict).
        """
        prompt = f"""
You are {self._name} ({self._role.value}) in the Werewolf debate phase.

{self._note}

How urgently do you want to speak next?
  8-10 : Strong, specific accusation you must voice immediately.
  4-7  : Useful observations worth sharing, no immediate urgency.
  0-3  : Little new to contribute right now.

Respond with ONLY a JSON object:
{{
  "bid": integer from 0 to 10,
  "reason": "one-line private rationale (<=15 words)"
}}
No extra text, no markdown, no code fences.
"""
        resp = self.call_model(prompt, max_tokens=100)

        try:
            bid = max(0, min(10, int(resp.get("bid", 5))))
        except (TypeError, ValueError):
            bid = 5
            resp["fallback_bid"] = "Defaulted to 5 — invalid model response."

        return bid, resp

    def vote(self, alive_players: List[str]) -> tuple[Optional[str], dict]:
        """Vote to eliminate a player during the day phase, or abstain.
        Returns (target_name or None, log_dict).
        """
        available = [p for p in alive_players if p != self._name]
        if not available:
            return None, {"error": "No available targets to vote for."}

        prompt = f"""
You are {self._name} ({self._role.value}). Win for your faction.
Cast a vote for who should be exiled. You may choose to skip voting if you are unsure.

{self._note}

Available targets: {", ".join(available)}

Respond with ONLY a JSON object:
{{
  "vote": "name of one player to exile, or 'None' to skip",
  "is_deceptive": true/false,
  "analysis": "private rationale (<=20 words)",
  "reasoning": "public explanation (<=20 words)"
}}
No extra text, no markdown, no code fences.
"""
        resp = self.call_model(prompt, max_tokens=200)
        target = resp.get("vote", "None")

        # 1. Handle explicit abstention
        if target == "None" or target is None:
            return None, resp

        # 2. If the target is NOT in the list, immediately make it None (skip)
        if target not in available:
            resp["fallback_vote"] = f"Forced None: Invalid vote '{target}' not in alive players."
            return None, resp

        # 3. Otherwise, return the valid target!
        return target, resp

    def debate(self) -> tuple[str, dict]:
        """Contribute a statement to the day debate.
        Returns (statement, log_dict).
        """

        prompt = f"""
You are {self._name} ({self._role.value}).
Win for your faction. Be assertive — avoid hedging.

{self._note}

Make a strong, decisive accusation or defence.

Respond with ONLY a JSON object:
{{
  "statement": "natural, decisive line (<=20 words)",
  "is_deceptive": true/false,
  "analysis": "private reasoning (<=20 words)"
}}
No extra text, no markdown, no code fences.
"""
        resp = self.call_model(prompt, max_tokens=200)
        message = resp.get("statement", "").strip()

        if not message:
            return "", {"error": "No valid statement generated."}

        return message, resp

    # ── LLM call ────────────────────────────────────────────────────────

    def call_model(self, prompt: str, max_tokens: int = 200, timeout: int = 15) -> dict:
        """Send a two-message request to the LLM.

        SystemMessage : self._setup_prompt = role + personality + strategy + rules.
                        Fixed for the player's lifetime; re-sent on EVERY call because the LLM has no memory between calls.
        HumanMessage  : prompt = game context + current task.

        Returns parsed JSON dict; falls back to {"raw": str} on parse failure.
        """
        messages = [
            SystemMessage(content=self._setup_prompt),
            HumanMessage(content=prompt),
        ]
        # Create a dictionary of kwargs so we can dynamically adjust for Google vs OpenAI
        kwargs = {"timeout": timeout}

        # Fix for the LangChain/Google kwarg bug
        if "GoogleGenerativeAI" not in type(self._model).__name__:
            # Only enforce max_tokens at the invoke level for OpenAI/Anthropic.
            # Gemini will naturally stop based on the prompt's word limits!
            kwargs["max_tokens"] = max_tokens

        resp = self._model.invoke(messages, **kwargs).content
        resp = resp.strip() if isinstance(resp, str) else resp

        result: dict = {}
        try:
            result = json.loads(resp)
        except json.JSONDecodeError:
            result = {"raw": resp}

        result.setdefault("_raw_response", resp)
        result.setdefault("_prompt", prompt)
        return result

    # ── Post-game: coach feedback ────────────────────────────────────────

    @property
    def _coach_feedback(self) -> str:
        """Read coach feedback from disk (written by Coach.run()).
        Returns empty string if file is missing — coaching was not run.
        Only meaningful for villager-side players.
        """
        path = self._feedback_path
        if not path.exists():
            return ""
        try:
            return path.read_text().strip()
        except Exception:
            return ""

    # ── Post-game: strategy update ───────────────────────────────────────

    def update_strategy(self) -> None:
        """Update [name]_strategy.txt after a game ends.

        Villager-side  (Villager, Seer, Guard, Witch)
        ───────────────────────────────────────────────────────────────────
          self_analyze  coaching  Sources used
          ────────────  ────────  ─────────────────────────────────────────
          True          True      self-analysis + coach feedback + current strategy
          True          False     self-analysis + current strategy
          False         True      coach feedback + current strategy
          False         False     no-op — nothing to learn from

        Wolf-side  (Werewolf)
        ───────────────────────────────────────────────────────────────────
          Always self-analyzes the full game.
          coaching and self_analyze flags are IGNORED — wolves have no coach and always review the game themselves regardless of experiment config.

        After writing, refreshes self._setup_prompt so the NEXT game immediately benefits from the updated strategy.
        """
        current_strategy = self._load_strategy()
        game_record      = self._game_summary

        if self._role in VILLAGER_SIDE:
            new_strategy = self._update_strategy_villager(
                current_strategy=current_strategy,
                game_record=game_record,
            )
        else:
            # Wolves: always self-analyze, never use coach
            new_strategy = self._update_strategy_wolf(
                current_strategy=current_strategy,
                game_record=game_record,
            )

        if new_strategy:
            self._write_strategy(new_strategy)

    def _update_strategy_villager(self, current_strategy: str, game_record: str) -> Optional[str]:
        """Build and execute the LLM prompt for villager-side strategy update.
        Returns the new strategy string, or None when both flags are False.
        """
        if not self._self_analyze and not self._coaching:
            return None

        sources: List[str] = []

        if self._self_analyze:
            sources.append(
                "=== Your Own Game Analysis ===\n"
                f"What happened this game from your point of view:\n{game_record}\n"
                "Reflect: what worked, what failed, what you should do differently."
            )

        if self._coaching:
            feedback = self._coach_feedback
            if feedback:
                sources.append(f"=== Coach Feedback ===\n{feedback}")

        if current_strategy:
            sources.append(f"=== Your Current Strategy ===\n{current_strategy}")

        context = "\n\n".join(sources)

        prompt = f"""
You are {self._name}, a {self._role.value}. The game has ended.

{context}

Win condition: ensure the Villagers eliminate all Werewolves.

Produce an updated strategy — clear, actionable behavioural rules for future games based on the roles, not the players' names.
Cover early-game, mid-game, and late-game. Discard rules that failed; keep what worked.

Respond with ONLY a JSON object:
{{
  "strategy": "updated rules as bullet points (<=120 words)",
  "reasoning": "main changes and why (<=30 words)"
}}
No extra text, no markdown, no code fences.
"""
        resp = self.call_model(prompt, max_tokens=500)
        return resp.get("strategy", "")

    def _update_strategy_wolf(self, current_strategy: str, game_record: str) -> Optional[str]:
        """Build and execute the LLM prompt for wolf strategy update.
        Wolves always self-analyze. No coaching, no flags — always runs.
        Returns the new strategy string.
        """
        sources: List[str] = [f"=== Full Game Record (your view) ===\n{game_record}"]

        if current_strategy:
            sources.append(f"=== Your Current Strategy ===\n{current_strategy}")

        context = "\n\n".join(sources)

        prompt = f"""
You are {self._name}, a Werewolf. The game has ended.

{context}

Win condition: eliminate enough Villagers so wolves equal or outnumber them, without being exiled during the day phase.

Analyse the full game: which deceptions worked, which accusations were dangerous, when to stay quiet vs. vocal, and how to avoid detection next time.

Produce an updated strategy — clear, actionable behavioural rules for future games.

Respond with ONLY a JSON object:
{{
  "strategy": "updated rules as bullet points (<=120 words)",
  "reasoning": "main changes and why (<=30 words)"
}}
No extra text, no markdown, no code fences.
"""
        resp = self.call_model(prompt, max_tokens=500)
        return resp.get("strategy", "")

    def _write_strategy(self, strategy: str) -> None:
        """Persist strategy to disk and refresh self._setup_prompt."""
        write_to_file(self._strategy_path, strategy)
        self._setup_prompt = self._build_setup_prompt(strategy)

    # ── Between-game reset ───────────────────────────────────────────────

    def reset_game_state(self, game_id: str, other_players: List[str], scenario: str = None) -> None:
        """Clear all intra-game state for a fresh game.
        Call from run.py at the start of each game instead of re-instantiating.
        Strategy (in _setup_prompt) is preserved across resets.

        Args:
            game_id:       New game identifier used in file names.
            other_players: Names of all other players in this new game.
            scenario:      Experiment scenario; updates _scenario if provided.
        """
        self._game_id = game_id
        if scenario is not None:
            self._scenario = scenario
        self._game_summary_entries = []
        self._is_alive = True
        self.init_suspicions(other_players)