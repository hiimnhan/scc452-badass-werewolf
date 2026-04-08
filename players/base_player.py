from collections import defaultdict
from enum import Enum
import json
from typing import List
from langchain_core.language_models import BaseChatModel
from abc import ABC
from pathlib import Path

from constants import COACH_FEEDBACK_FILENAME


class Role(Enum):
    VILLAGER = "Villager"
    WEREWOLF = "Werewolf"
    SEER = "Seer"
    GUARD = "Guard"
    WITCH = "Witch"


class BasePlayer(ABC):
    def __init__(
        self,
        name: str,
        role: Role,
        model: BaseChatModel,
        is_alive: bool = True,
        system_prompt: str = "",
        personality: str = "",
    ) -> None:
        self._name = name
        self._role = role
        self._model = model
        self._is_alive = is_alive
        self._current_game_notes = []  # Player's private _current_game or thoughts current game
        self._self_reflections = (
            ""  # Actionable guidelines for the player based on feedback
        )
        self._suspicions = defaultdict(float)  # Suspicion scores for other players
        self._personality = personality

        self.setup_prompt = ", ".join(
            [
                system_prompt.format(name=name),
                f"Your personality is: {personality}" if personality else "",
            ]
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name}, role={self._role.value}, alive={self._is_alive})"

    def self_reflect(self, experiment):
        """

        Based on experiment being conducted,
        Get coach feedback (or not) + last game reflection to update player's own
        """
        if (
            experiment in []
        ):  # TODO: in these scenarios we dont use coach feedback. Need to define these.
            coach_feeback = "No feedback provided"  # FIXME:
        coach_feeback, err = self._get_coach_feeback()
        if err:
            return err

        if (
            experiment in []
        ):  # TODO: in these scenarios we dont use last game reflection. Need to define these.
            last_game_reflection = " No last game reflection provided"  # FIXME:

        last_game_reflection, err = self._get_last_game_reflection()
        if err:
            return err

        prompt = f"""
        Last game I learned: {last_game_reflection}. My coach told me: {coach_feeback}.
Think step by step: analyse my role's win condition, turn my past lessons and coach's feedback into behavioural rules, then give me a clear strategy to carry through the entire game — how to act early, mid, and late game.
        Respond with ONLY a single JSON object using these exact keys and constraints:
        {{
        "strategy": "strategies to play in the whole game",
        "reasoning": "brief explain explanation "
        }}
        No extra text, no markdown, no code fences.
        """
        resp = self.call_model(prompt)
        strategy = resp.get("strategy", "")
        self._self_reflections = strategy

    def _add_note(self, note: str) -> None:
        self._current_game_notes.append(note)

    def call_model(self, prompt: str, max_tokens: int = 200, timeout: int = 15) -> dict:
        """Calls the LLM with the given prompt. Output parsed JSON including raw response and prompt"""

        prompt = f"{self.setup_prompt}\n{prompt}"  # NOTE: ADD SYSTEM PROMPT TO THE BEGINNING OF THE PROMPT
        resp = self._model.invoke(
            prompt, max_tokens=max_tokens, timeout=timeout
        ).content
        resp = resp.strip() if isinstance(resp, str) else resp

        result = {}
        try:
            result = json.loads(resp)
        except json.JSONDecodeError:
            result = {"raw": resp}

        result.setdefault("_raw_response", resp)
        result.setdefault("_prompt", prompt)

        return result

    def vote(self, alive_players: list[str]) -> tuple[str, dict]:
        """Votes to eliminate a target player during the day phase
        Return target player name and log
        """
        available_targets = [p for p in alive_players if p != self._name]

        if not available_targets:
            return "", {"error": "No available targets to vote for."}
        prompt = f"""
        You are {self._name} ({self._role}). Your sole objective is to win for your faction.
        Based on the debate and observations, cast a decisive vote for who you think should be exiled.

        Here are some action guidelines that you can consider: {self._self_reflections if self._self_reflections else "No specific guidelines at the moment."}

        Here are your current suspicions on other players: {", ".join([f"{p} ({s:.2f})" for p, s in self._suspicions.items()]) if self._suspicions else "No specific suspicions at the moment."}

        Here are some notes you have taken during the game: {", ".join(self._current_game_notes) if self._current_game_notes else "No notes taken yet."}

        Respond with ONLY a single JSON object using these exact keys and constraints:
        {{
        "vote": "name of one player you vote to exile (one word)",
        "is_deceptive": true/false,
        "analysis": "private, terse rationale (<=20 words)",
        "reasoning": "brief public explanation (<=20 words)"
        }}
        No extra text, no markdown, no code fences.
        """
        resp = self.call_model(prompt)
        target = resp.get("vote", "")

        if target not in available_targets:
            # extract valid target from raw response if possible
            if "raw" in resp:
                raw = resp["raw"]
                for t in available_targets:
                    if t in raw:
                        target = t
                        break
            return "", {
                "error": f"Invalid target '{target}'. Must choose from: {available_targets}"
            }

        self._add_note(
            f"Voted to eliminate {target}. Reason: {resp.get('analysis', 'No analysis provided.')}"
        )

        return target, resp

    def debate(self, dialogue_history: List[dict]) -> tuple[str, dict]:
        """Engages in debate with other players based on dialogue history and guideline for the role.
        returns message to contribute to debate and log
        """
        history = "\n".join([f"{s}: {t}" for s, t in dialogue_history])

        # Add context about the game state
        context = f"You are {self._name} ({self._role}). This is a werewolf game where villagers try to find werewolves and werewolves try to blend in. Your sole objective is to win for your faction. Be assertive and competitive. Avoid hedging or niceties."
        prompt = f"""
{context}

Dialogue history so far:
{history if history else "No previous dialogue."}

Here are some action guidelines that you can consider: {self._self_reflections if self._self_reflections else "No specific guidelines at the moment."}

Here are your current suspicions on other players: {", ".join([f"{p} ({s:.2f})" for p, s in self._suspicions.items()]) if self._suspicions else "No specific suspicions at the moment."}

Here are some notes you have taken during the game: {", ".join(self._current_game_notes) if self._current_game_notes else "No notes taken yet."}

You must contribute to the debate with a strong, actionable accusation or defense.
Respond with ONLY a single JSON object using these exact keys and constraints:
{{
  "statement": "a natural, decisive line (<=20 words) that advances your win condition",
  "is_deceptive": true/false,
  "analysis": "your private, terse reasoning (<=20 words)"
}}

No extra text, no markdown, no code fences.
"""
        resp = self.call_model(prompt)
        message = resp.get("statement", "")

        if not message or message.strip() == "":
            if (
                "raw" in resp
                and isinstance(resp["raw"], str)
                and resp["raw"].strip() != ""
            ):
                raw = resp["raw"].strip()
                if "statement" in raw.lower():
                    # try to extract message after "message:" if present
                    import re

                    match = re.search(r"message\s*:\s*(.+)", raw, re.IGNORECASE)
                    if match:
                        message = match.group(1).strip()

            return "", {"error": "No valid message generated for debate contribution."}

        self._add_note(f"Contributed to debate: {message}")

        return message, resp

    def _get_coach_feeback(self) -> tuple[str, dict | None]:
        """
        Get coach feedback
        return (feeback, error)
        """
        base = Path(__file__).parent
        filepath = (base / f"../feedbacks/{COACH_FEEDBACK_FILENAME}").resolve()

        try:
            with open(filepath, "r") as f:
                feedback = f.read()
        except FileNotFoundError:
            return "", {
                "error": f"No feedback file found for role {self._role.value} at {filepath}"
            }
        except PermissionError:
            return "", {
                "error": f"Permission denied when trying to read feedback file for role {self._role.value} at {filepath}"
            }
        except Exception as e:
            return "", {"error": e}
        return feedback, None

    def _get_last_game_reflection(self) -> tuple[str, dict | None]:
        """
        Get last game reflection
        return (reflection, error)
        """
        base = Path(__file__).parent
        filepath = (base / f"../strategies/{self._name}").resolve()

        try:
            with open(filepath, "r") as f:
                reflection = f.read()
        except FileNotFoundError:
            return "", {
                "error": f"No feedback file found for role {self._role.value} at {filepath}"
            }
        except PermissionError:
            return "", {
                "error": f"Permission denied when trying to read feedback file for role {self._role.value} at {filepath}"
            }
        except Exception as e:
            return "", {"error": e}
        return reflection, None

    def detect_deception(
        self,
        speaker_name: str,
        statement: str,
        context: str = "",
        history: List[dict] = [],
    ) -> dict:
        """
        Detect deception from last speaker and update suspicious scores
        """
        history_text = ""
        if history:
            recent_history = history[-3:]
            history_items = []
            for h in recent_history:
                deceptive = "deceptive" if h.get("is_deceptive", 0) == 1 else "truthful"
                history_items.append(
                    f"- {h.get('statement', 'Unknown')}: classified as {deceptive}"
                )
            history_text = f"\nPrevious statements from {speaker_name}:\n" + "\n".join(
                history_items
            )
        prompt = f"""
Another player, {speaker_name}, just made this statement: "{statement}"
Context: {context}{history_text}

Please analyze {speaker_name}'s statement using Chain of Thought reasoning. 
Think step by step to analyze deception:
1. Likely intentions behind the statement
2. Consistency with prior behavior/claims
3. Contradictions, unusual phrasing, or suspicious elements
4. Truthfulness given game knowledge
5. Consider role possibilities and motivations

Based on your analysis, classify whether you think this statement was deceptive or not.

Respond in this exact JSON format:
{{
    "chain_of_thought": "Your step-by-step reasoning about {speaker_name}'s statement",
    "is_deceptive": 0 or 1,
    "confidence": 0.0 to 1.0,
    "deception_type": "none" | "omission" | "distortion" | "fabrication" | "misdirection",
    "reasoning": "Brief explanation of your classification",
    "suspicion_level": 0.0 to 1.0
}}
No extra text, no markdown, no code fences.
"""
        resp = self.call_model(prompt)
        self._suspicions[speaker_name] = resp.get(
            "suspicion_level", self._suspicions[speaker_name]
        )
        return resp
