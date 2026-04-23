from typing import List
from players.base_player import BasePlayer, Role
from prompt import GUARD_PROMPT_TEMPLATE, GUARD_PROTECT_PROMPT_TEMPLATE
from langchain_core.language_models import BaseChatModel
from constants import MAX_RETRIES


class Guard(BasePlayer):
    def __init__(
        self,
        name: str,
        model: BaseChatModel,
        game_id: str = "",
        scenario: str = "baseline",
        role: Role = Role.GUARD,
        system_prompt: str = GUARD_PROMPT_TEMPLATE,
        personality: str = "",
    ):
        super().__init__(
            name=name,
            role=role,
            model=model,
            game_id=game_id,
            scenario=scenario,
            system_prompt=system_prompt,
            personality=personality,
        )
        self.last_guarded_player = None

    def protect(self, alive_players: list[str], round_num: int) -> tuple[str, str, dict]:
        if not alive_players:
            return "None", "No alive players provided.", {"error": "No alive players provided."}

        available_targets = [player for player in alive_players if player != self.last_guarded_player]
        if not available_targets:
            return "None", "No valid targets to protect.", {"error": "No valid targets to protect."}

        prompt = GUARD_PROTECT_PROMPT_TEMPLATE.format(
            name=self._name,
            note=self._note,
            list_player=", ".join(available_targets)
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

        # Validate that the target is actually in the available targets
        if target not in available_targets:
            # 1st Fallback: Try to extract a valid name from the raw response
            if "raw" in resp:
                raw_response = resp["raw"]
                for player in available_targets:
                    if player in raw_response:
                        target = player
                        # We found a name, but we still need a safe analysis string
                        analysis = "Extracted target from raw text due to JSON failure."
                        resp["target"] = target
                        resp["analysis"] = analysis
                        break

            # 2nd Fallback: If still no valid target, pick a random valid target
            if target not in available_targets:
                import random
                print(f"Invalid target '{target}' received. Available targets: {available_targets}.") 
                
                target = random.choice(available_targets)
                analysis = "Forced random target due to completely invalid LLM response."
                
                # Sync the dictionary with our new variables
                resp["target"] = target
                resp["analysis"] = analysis
                resp["fallback"] = analysis

        # Record the final, validated action
        self.record_own_action(
            round_num, "Night", f"Protected {target}. Reason: {analysis}"
        )
        self.last_guarded_player = target
        
        return target, analysis, resp