from typing import List

from players.base_player import BasePlayer, Role
from prompt import GUARD_PROMPT_TEMPLATE, GUARD_PROTECT_PROMPT_TEMPLATE
from langchain_core.language_models import BaseChatModel


class Guard(BasePlayer):
    def __init__(
        self,
        name: str,
        model: BaseChatModel,
        role: Role = Role.GUARD,
        system_prompt: str = GUARD_PROMPT_TEMPLATE,
    ):
        super().__init__(name=name, role=role, model=model, system_prompt=system_prompt)
        self.last_guarded_player = None

    def protect(self, alive_players: List[str] = None) -> tuple[str, dict]:
        # Implement the logic to protect a player
        # input: alive players (list), output: target player to protect (str), log (dict)
        # how to use: call this method, it will return the name of the player to protect
        if alive_players is None:
            return "", {"error": "No alive players provided."}

        available_targets = [
            player for player in alive_players if player != self.last_guarded_player
        ]
        if not available_targets:
            return "", {"error": "No valid targets to protect."}

        prompt = GUARD_PROTECT_PROMPT_TEMPLATE.format(
            name=self._name, list_player=", ".join(available_targets)
        )
        # print(prompt)  # Debug only
        resp = self.call_model(prompt, max_tokens=300)
        target = resp.get("target", "")
        
        # Validate that the target is actually in the available targets
        if target not in available_targets:
            # If invalid target, try to extract a valid name from the response
            # prevent model doesn't reply expected JSON format
            if "raw" in resp:
                # print(f"Raw response for debugging: {resp}")  # Debug only
                raw_response = resp["raw"]
                for player in available_targets:
                    if player in raw_response:
                        target = player
                        break
            
            # If still no valid target, pick the first available target
            if target not in available_targets:
                print(f"Invalid target '{target}' received. Available targets: {available_targets}.")  # Debug only
                target = available_targets[0]
                resp["target"] = target
                resp["fallback"] = "Used first available target due to invalid response"

        self._add_note(
            f"Guard protect {target}. Reason: {resp.get('analysis', 'No analysis provided.')}"
        )
        self.last_guarded_player = target
        return target, resp