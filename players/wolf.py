from players.base_player import BasePlayer, Role
from typing import List
from prompt import WEREWOLF_PROMPT_TEMPLATE, WEREWOLF_ELIMINATE_PROMPT_TEMPLATE, WEREWOLF_DEBATE_PROMPT_TEMPLATE
import json
import re

class Wolf(BasePlayer):
    def __init__(self,
        name,
        model,
        game_id: str = "",
        scenario: str = "baseline",
        is_alive=True,
        personality=""
    ):
        super().__init__(name=name, role=Role.WEREWOLF, model=model, game_id=game_id, scenario=scenario, is_alive=is_alive, system_prompt=WEREWOLF_PROMPT_TEMPLATE, personality=personality)
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
        
        if "statement" not in response:
            match = re.search(r"\{.*\}", response.get("_raw_response", ""), re.DOTALL)
            if match: response.update(json.loads(match.group()))
        
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
        
        if "target" not in response:
            match = re.search(r"\{.*\}", response.get("_raw_response", ""), re.DOTALL)
            if match: response.update(json.loads(match.group()))
        
        target_eliminate = response.get("target", "")
        
        # In case if AI picks not existed name, proceed with picking the first valid one
        if target_eliminate not in targets:
            target_eliminate = targets[0] if targets else ""

        self.record_own_action(round_num, "Night", f"Night Action: Eliminating {target_eliminate}.")
        return target_eliminate, response
    
    def update_suspicion_from_statement(self, speaker_name: str, statement: str) -> dict:
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
        
        if "updates" not in resp:
            match = re.search(r'\{.*\}', resp.get("_raw_response", ""), re.DOTALL)
            if match:
                try: resp.update(json.loads(match.group()))
                except: pass
    
        updates = resp.get("updates", {})
    
        for player, data in updates.items():
            if player in self._suspicion:
                try:
                    new_score = float(data.get("score", self._suspicion[player]["score"]))
                except:
                    new_score = self._suspicion[player]["score"]
                self._suspicion[player] = {
                    "score": new_score,
                    "reason": data.get("reason", self._suspicion[player]["reason"])
                }
        return resp
    
    def update_suspicion_from_vote(self, current_round_vote_logs: list[str], exiled_player: str, game_status: str) -> dict:
        """Override to evaluate DANGER to the pack based on voting behavior."""
        voting_summary = "\n".join(current_round_vote_logs)
        voting_summary += f"\n\nOutcome: {exiled_player} exiled. {game_status}"
        
        prompt = f"""
    You are {self._name} (Werewolf). 
    The daily vote just concluded. Here are the results:
    {voting_summary}

    {self._note}

    Analyse these voting patterns. Identify DANGER to the Werewolves:
    1. Who is successfully leading votes against your teammates?
    2. Who is consistently voting with the majority against the pack?
    3. Who is the most 'dangerous' villager based on their voting accuracy?

    Respond with ONLY a JSON object:
    {{
    "chain_of_thought": "voting analysis (<=40 words)",
    "updates": {{
        "PlayerName": {{"score": 0.0 to 1.0, "reason": "danger analysis (<=40 words)"}}
    }}
    }}
    """
        resp = self.call_model(prompt, max_tokens=800)
        
        if "updates" not in resp:
            match = re.search(r"\{.*\}", resp.get("_raw_response", ""), re.DOTALL)
            if match:
                try: resp.update(json.loads(match.group()))
                except: pass

        updates = resp.get("updates", {})
        for player, data in updates.items():
            if player in self._suspicion:
                try:
                    new_score = float(data.get("score", self._suspicion[player]["score"]))
                except:
                    new_score = self._suspicion[player]["score"]
                self._suspicion[player] = {"score": new_score, "reason": data.get("reason", self._suspicion[player]["reason"])}
        return resp
