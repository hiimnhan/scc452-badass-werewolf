from players.base_player import BasePlayer, Role
from typing import List
from prompt import WEREWOLF_PROMPT_TEMPLATE, WEREWOLF_ELIMINATE_PROMPT_TEMPLATE, WEREWOLF_DEBATE_PROMPT_TEMPLATE
import re

class Wolf(BasePlayer):
    def __init__(self,
                    name,
                    model,
                    is_alive=True,
                    personality=""
                 ):
          super().__init__(name, Role.WEREWOLF, model, is_alive, WEREWOLF_PROMPT_TEMPLATE, personality)
          self._teammates = []

    def wolf_debate(self, alive_players: List[str], other_wolves: List[str], dialogue_history: List[dict]) -> tuple[str, dict]:
        """Private conversation between wolves for picking a target to eliminate, similar to voting logic that picks a name to eliminate"""
        self._teammates = other_wolves
        target = [p for p in alive_players if p not in other_wolves]
        history_conv = "\n".join([f"{s['player']}: {s['text']}" for s in dialogue_history])
        
        prompt = WEREWOLF_DEBATE_PROMPT_TEMPLATE.format(
                name=self._name,
                teammates = ", ".join(other_wolves),
                target_villagers = ", ".join(target),
                dialogue_history=history_conv  if history_conv else "There's no discussion yet.",
                self_reflections=self._self_reflections,
                notes=", ".join(self._current_game_notes)
            )
        response = self.call_model(prompt)
        message = response.get("statement", "")
        
        if not message or message.strip() == "":
            if "raw" in response and isinstance(response["raw"], str) and response["raw"].strip() != "":
                raw = response["raw"].strip()
                match = re.search(r"statement\s*:\s*(.+)", raw, re.IGNORECASE)
                if match: 
                    message = match.group(1).strip()
            
            if not message:
                return "", {"error": "No valid message generated for wolf debate."}

        self._add_note(f"Wolf Chat Contribution: {message}")
        return message, response

    def eliminate(self, alive_players: List[str]) -> tuple[str, dict]:
        """The final decision on who dies tonight."""
        available_targets = [p for p in alive_players if p != self._name and p not in self._teammates]
        
        prompt = WEREWOLF_ELIMINATE_PROMPT_TEMPLATE.format(
            name=self._name,
            target_pool=", ".join(available_targets),
            self_reflections=self._self_reflections,
            notes=", ".join(self._current_game_notes)
        )
        response = self.call_model(prompt)
        target_eliminate = response.get("target", "")
        
        # In case if AI picks not existed name, proceed with picking the first valid one
        if target_eliminate not in available_targets:
            target_eliminate = available_targets[0] if available_targets else ""

        self._add_note(f"Night Action: Eliminating {target_eliminate}.")
        return target_eliminate, response
    
