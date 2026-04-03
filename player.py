from collections import defaultdict
from enum import Enum
import json
from typing import List, override
from langchain_core.language_models import BaseChatModel


class Role(Enum):
    VILLAGER = "Villager"
    WEREWOLF = "Werewolf"
    SEER = "Seer"
    DOCTOR = "Doctor"


class Player:
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
        self._notes = []  # Player's private notes or thoughts
        self._suspicions = defaultdict(float)  # Suspicion scores for other players
        self._personality = personality

        self.setup_prompt = system_prompt.format(name)

    @override
    def __repr__(self) -> str:
        return f"Player(name={self._name}, role={self._role.value}, alive={self._is_alive})"

    def _add_note(self, note: str) -> None:
        self._notes.append(note)

    def update_suspicion(self, player_name: str, suspicion_score: float) -> None:
        self._suspicions[player_name] = suspicion_score

    def call_model(self, prompt: str, max_tokens: int = 200, timeout: int = 15) -> dict:
        """Calls the LLM with the given prompt. Output parsed JSON including raw response and prompt"""
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

    def eliminate(self, alive_players: List[str]) -> tuple[str, dict]:
        """Werewolf eliminates a target player
        Return target player name and log
        """
        available_targets = [p for p in alive_players if p != self._name]

        if not available_targets:
            return "", {"error": "No available targets to eliminate."}

        prompt = ""
        """
        TODO - prompt should return JSON format
        {{
            "target": "<name of target player to eliminate>",
            "analysis": "<optional analysis or reasoning for the choice> <= 20 words"
        }}
        """
        resp = self.call_model(prompt)
        target = resp.get("target", "")

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
            f"Chose to eliminate {target}. Reason: {resp.get('analysis', 'No analysis provided.')}"
        )

        return target, resp

    def save(self, alive_players: List[str]) -> tuple[str, dict]:
        """Doctor saves a target player
        Return target player name and log
        """
        available_targets = [p for p in alive_players if p != self._name]

        if not available_targets:
            return "", {"error": "No available targets to save."}

        prompt = ""
        """
        TODO - prompt should return JSON format
        {{
            "target": "<name of target player to save>",
            "analysis": "<optional analysis or reasoning for the choice> <= 20 words"
        }}
        """
        resp = self.call_model(prompt)
        target = resp.get("target", "")

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
            f"Chose to save {target}. Reason: {resp.get('analysis', 'No analysis provided.')}"
        )

        return target, resp

    def unmask(self, alive_players: List[str]) -> tuple[str, dict]:
        """
        Seer reveals a target player's role
        Return target player name and log
        """
        available_targets = [p for p in alive_players if p != self._name]
        if not available_targets:
            return "", {"error": "No available targets to reveal."}

        prompt = ""
        """
        TODO - prompt should return JSON format
        {{
            "target": "<name of target player to reveal>",
            "analysis": "<optional analysis or reasoning for the choice> <= 20 words"
        }}
        """
        resp = self.call_model(prompt)
        target = resp.get("target", "")

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
            f"Chose to reveal {target}. Reason: {resp.get('analysis', 'No analysis provided.')}"
        )

        return target, resp

    def update_revealed_role(self, target: str, role: Role) -> None:
        """Updates player's notes with revealed role information"""
        self._add_note(f"Learned that {target} is a {role.value}.")

    def debate(self, dialogue_history: List[dict]) -> tuple[str, dict]:
        """Engages in debate with other players based on dialogue history
        returns message to contribute to debate and log
        """
        prompt = ""
        """
        TODO - prompt should return JSON format
        {{
            "message": "<message to contribute to the debate>",
            "analysis": "<optional analysis or reasoning for the contribution> <= 20 words"
        }}
        """
        resp = self.call_model(prompt)
        message = resp.get("message", "")

        if not message or message.strip() == "":
            if (
                "raw" in resp
                and isinstance(resp["raw"], str)
                and resp["raw"].strip() != ""
            ):
                raw = resp["raw"].strip()
                if "message" in raw.lower():
                    # try to extract message after "message:" if present
                    import re

                    match = re.search(r"message\s*:\s*(.+)", raw, re.IGNORECASE)
                    if match:
                        message = match.group(1).strip()

            # TODO - should use a default message
            return "", {"error": "No valid message generated for debate contribution."}

        self._add_note(f"Contributed to debate: {message}")

        return message, resp

    def vote(self, alive_players: List[str]) -> tuple[str, dict]:
        """Votes to eliminate a target player during the day phase
        Return target player name and log
        """
        available_targets = [p for p in alive_players if p != self._name]

        if not available_targets:
            return "", {"error": "No available targets to vote for."}

        prompt = ""
        """
        TODO - prompt should return JSON format
        {{
            "target": "<name of target player to vote for elimination>",
            "analysis": "<optional analysis or reasoning for the choice> <= 20 words"
            "reasoning": "<optional detailed reasoning for the choice for public debate>""
        }}
        """
        resp = self.call_model(prompt)
        target = resp.get("target", "")

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
