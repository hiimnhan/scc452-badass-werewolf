from players.base_player import BasePlayer, Role
from prompt import WEREWOLF_PROMPT_TEMPLATE, WEREWOLF_ELIMINATE_PROMPT_TEMPLATE
from constants import MAX_RETRIES

class Wolf(BasePlayer):
    def __init__(
        self,
        name,
        model,
        game_id: str = "",
        scenario: str = "baseline",
        role: Role = Role.WEREWOLF,
        is_alive=True,
        personality="",
    ):
        super().__init__(
            name=name,
            role=role,
            model=model,
            game_id=game_id,
            scenario=scenario,
            is_alive=is_alive,
            system_prompt=WEREWOLF_PROMPT_TEMPLATE,
            personality=personality,
        )

    def eliminate(
        self, alive_players: list[str], wolf_teammates: list[str], dialogue_history: list, round_num: int
    ) -> tuple[str, str, str, dict]:
        """The final decision on who dies tonight."""

        # 1. Filter teammates to see who is actually alive
        alive_teammates = [p for p in wolf_teammates if p in alive_players and p != self._name]

        # 2. Determine available targets (anyone alive who isn't a wolf)
        targets = [p for p in alive_players if p != self._name and p not in wolf_teammates]

        # 3. Format the teammates string dynamically
        if not alive_teammates:
            teammates_str = "None. You are the last wolf standing."
        else:
            teammates_str = ", ".join(alive_teammates)

        # 4. Format the dialogue history into "Speaker: Statement"
        formatted_lines = []
        for entry in dialogue_history:
            # Assuming entry is [round_num, speaker_name, statement, target, analysis]
            speaker_name = entry[1]
            statement = entry[2]
            formatted_lines.append(f"{speaker_name}: {statement}")

        formatted_dialogue = "\n".join(formatted_lines) if formatted_lines else "No debate occurred."

        # 5. Format the prompt
        prompt = WEREWOLF_ELIMINATE_PROMPT_TEMPLATE.format(
            name=self._name,
            role=self._role.value,
            teammates=teammates_str,
            target_pool=", ".join(targets),
            note=self._note,
            dialogue_history=formatted_dialogue,
        )

        # 6. Call the LLM safely with a retry loop
        target = ""
        statement = ""
        analysis = "Default analysis: LLM failed to provide reasoning."

        for attempt in range(MAX_RETRIES):
            resp = self.call_model(prompt, max_tokens=200)
            
            # 7. Check if ALL required keys are in the response
            if "target" in resp and "statement" in resp and "analysis" in resp:
                target = resp["target"]
                statement = resp["statement"]
                analysis = resp["analysis"]
                break # We got everything we need, exit the loop!
            else:
                print(f"Warning: {self._name} ({self._role.value}) failed to generate valid elimination JSON (Attempt {attempt + 1}/{MAX_RETRIES})")

        # 8. VALIDATION: Handle the case where the target is empty, None, or an invalid hallucinated name
        if target not in targets:
            import random
            target = random.choice(targets) if targets else "None" # Fallback to a valid string
            statement = "Failed to choose a target. Randomly select a target."
            analysis = "Failed to choose a target. Randomly select a target."
            resp["fallback_target"] = "Forced random target: Invalid or missing target chosen by LLM."
            
        # 9. RECORD: Log the final, validated action (moved below the validation!)
        self.record_own_action(
            round_num, "Night", f"Debate: Targeted {target}. Reason: {analysis}"
        )
            
        return target, statement, analysis, resp