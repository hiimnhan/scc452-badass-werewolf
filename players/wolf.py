from players.base_player import BasePlayer, Role
from typing import List
from prompt import WEREWOLF_PROMPT_TEMPLATE, WEREWOLF_ELIMINATE_PROMPT_TEMPLATE, WEREWOLF_DEBATE_PROMPT_TEMPLATE
import re

class Wolf(BasePlayer):
    def __init__(self,
                    name,
                    model,
                    game_id,
                    is_alive=True,
                    personality=""
                 ):
          super().__init__(name, Role.WEREWOLF, model,game_id , is_alive, WEREWOLF_PROMPT_TEMPLATE, personality)
          self._teammates = []

    def wolf_debate(self, alive_players: List[str], other_wolves: List[str], dialogue_history: List[dict], round_num: int) -> tuple[str, dict]:
        """Private conversation between wolves for picking a target to eliminate, similar to voting logic that picks a name to eliminate"""
        self._teammates = other_wolves
        target = [p for p in alive_players if p not in other_wolves and p != self._name]
        history_conv = "\n".join([f"{s}: {t}" for s, t in dialogue_history])
        
        prompt = WEREWOLF_DEBATE_PROMPT_TEMPLATE.format(
                name=self._name,
                teammates = ", ".join(other_wolves),
                target_pool = ", ".join(target),
                dialogue_history=history_conv  if history_conv else "There's no discussion yet.",
                note=self._note
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

        self.record_own_action(round_num, "Night", f"Wolf Chat: {message}")
        return message, response

    def eliminate(self, alive_players: List[str], round_num: int) -> tuple[str, dict]:
        """The final decision on who dies tonight."""
        targets = [p for p in alive_players if p != self._name and p not in self._teammates]
        
        prompt = WEREWOLF_ELIMINATE_PROMPT_TEMPLATE.format(
            name=self._name,
            target_pool=", ".join(targets),
            note=self._note
        )
        response = self.call_model(prompt)
        target_eliminate = response.get("target", "")
        
        # In case if AI picks not existed name, proceed with picking the first valid one
        if target_eliminate not in targets:
            target_eliminate = targets[0] if targets else ""

        self.record_own_action(round_num, "Night", f"Night Action: Eliminating {target_eliminate}.")
        return target_eliminate, response
    
    def update_suspicion(self, speaker_name: str, statement: str) -> dict:
        """To evaluate danger to the wolf-side."""
        prompt = f"""
    You are {self._name} (Werewolf). 
    {speaker_name} just said: "{statement}"

    {self._note}

    Evaluate the DANGER this player poses to the Werewolves. 
    High Danger (1.0) = Likely a Seer/Guard or a highly persuasive villager.
    Low Danger (0.0) = Easy to manipulate or quiet.

    Respond with ONLY a JSON object:
    {{
    "chain_of_thought": "strategic analysis (<=40 words)",
    "updates": {{
        "{speaker_name}": {{"score": 0.0 to 1.0, "reason": "danger analysis (<=40 words)"}}
    }}
    }}
    """
        resp = self.call_model(prompt, max_tokens=800)
        
        updates = resp.get("updates", {})
        for player, data in updates.items():
            if player in self._suspicion:
                self._suspicion[player] = {
                    "score": float(data.get("score", self._suspicion[player]["score"])),
                    "reason": data.get("reason", self._suspicion[player]["reason"])
                }
        return resp
    
