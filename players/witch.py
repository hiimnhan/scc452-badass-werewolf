from typing import List, Optional, Tuple
from langchain_core.language_models import BaseChatModel
from players.base_player import BasePlayer, Role
from prompt import WITCH_PROMPT_TEMPLATE, WITCH_SAVE_OR_POISON_PROMPT_TEMPLATE
from constants import MAX_RETRIES


class Witch(BasePlayer):
    """
    The Witch has two one-time use potions:
    - Save Potion: cancels the wolf kill on their target.
    - Poison Potion: eliminates any player, bypassing the Guard.

    Information rules:
    - The Witch sees the wolf target ONLY while her Save potion is available.
    - Once the Save potion is spent, the wolf target is hidden from her.
    """

    def __init__(
        self,
        name: str,
        model: BaseChatModel,
        game_id: str = "",
        scenario: str = "baseline",
        role: Role = Role.WITCH,
        is_alive: bool = True,
        system_prompt: str = WITCH_PROMPT_TEMPLATE,
        personality: str = "",
    ) -> None:
        super().__init__(
            name=name,
            role=role,
            model=model,
            game_id=game_id,
            scenario=scenario,
            is_alive=is_alive,
            system_prompt=system_prompt,
            personality=personality,
        )
        self._save_available: bool = True
        self._poison_available: bool = True

    def has_save_potion(self) -> bool:
        return self._save_available

    def has_poison_potion(self) -> bool:
        return self._poison_available

    def has_any_potion(self) -> bool:
        """Returns True if the Witch has at least one potion left."""
        return self.has_save_potion() or self.has_poison_potion()

    def save_or_poison(self, targeted_player_by_wolves: Optional[str], alive_players: List[str], round_num: int) -> Tuple[dict, str, dict]:
        """Night action: decide whether to use the save and/or poison potion.

        Returns (actions_dict, log_string, raw_response_dict).
        actions_dict format: {"use_save_potion": bool, "poison_target": str | None}
        """
        
        if not self._is_alive:
            return {"use_save_potion": False, "poison_target": None}, "Witch is dead.", {"error": "Witch is dead."}

        if not alive_players:
            return {"use_save_potion": False, "poison_target": None}, "No alive players.", {"error": "No alive players provided."}

        # The Witch only learns the wolf target while her Save potion is available.
        target_display = targeted_player_by_wolves if (self._save_available and targeted_player_by_wolves) else "None"
        alive_players_except_witch = [player for player in alive_players if player != self._name]
        alive_display = ", ".join(alive_players_except_witch) if self._poison_available else "None"

        prompt = WITCH_SAVE_OR_POISON_PROMPT_TEMPLATE.format(
            name=self._name,
            save_available=self._save_available,
            poison_available=self._poison_available,
            targeted_player=target_display,
            alive_players=alive_display,
            note=self._note,
        )

        directives_block = self._format_directives_block("NIGHT")
        if directives_block:
            prompt = f"{directives_block}\n\n{prompt}"

        # ---------------------------------------------------------
        # Safely Call the LLM (Max Retries, NO Raw Text Parsing)
        # ---------------------------------------------------------
        use_save = False
        poison_target = "None"
        save_reason = "Default: LLM failed to provide save reasoning."
        poison_reason = "Default: LLM failed to provide poison reasoning."
        resp = {}

        for attempt in range(MAX_RETRIES):
            resp = self.call_model(prompt, max_tokens=300)
            
            # Check if all 4 expected keys exist
            expected_keys = ["use_save_potion", "poison_target", "save_analysis", "poison_analysis"]
            if all(key in resp for key in expected_keys):
                
                # Safely parse the boolean (handles string "true" vs boolean True)
                raw_save = resp["use_save_potion"]
                if isinstance(raw_save, str):
                    use_save = raw_save.strip().lower() == "true"
                else:
                    use_save = bool(raw_save)
                    
                # Extract the rest safely
                poison_target = str(resp["poison_target"]).strip()
                save_reason = resp["save_analysis"]
                poison_reason = resp["poison_analysis"]
                break # Success! Exit the loop.
            else:
                print(f"Warning: {self._name} ({self._role.value}) failed to generate valid Witch JSON (Attempt {attempt + 1}/{MAX_RETRIES})")

        # ---------------------------------------------------------
        # Validation & Fallbacks
        # ---------------------------------------------------------

        # 1. Validate Save Potion
        if use_save and (not self._save_available or not targeted_player_by_wolves):
            use_save = False
            resp["fallback_save"] = "Forced False: Save potion unavailable or no active target."

        # 2. Validate Poison Potion (Strict Validation)
        if poison_target.lower() == "none" or poison_target == "":
            poison_target = None

        if poison_target and (not self._poison_available or poison_target not in alive_players_except_witch):
            poison_target = None
            resp["fallback_poison"] = "Forced None: Poison unavailable, target is invalid/dead, or LLM failed formatting."

        # ------------------------------------------------------------------
        # State update & own-action logging
        # ------------------------------------------------------------------
        
        if use_save:
            self._save_available = False
            self.record_own_action(
                round_num, "Night", f"Used SAVE potion on {targeted_player_by_wolves}. Reason: {save_reason}."
            )

        if poison_target:
            self._poison_available = False
            self.record_own_action(
                round_num, "Night", f"Used POISON potion on {poison_target}. Reason: {poison_reason}."
            )

        if not use_save and not poison_target:
            self.record_own_action(round_num, "Night", "Used no potions tonight.")

        log_string = f"Save reason: {save_reason}\nPoison reason: {poison_reason}"

        applicable = self._filter_directives_for_phase("NIGHT")
        applied_id = applicable[0].get("id") if applicable else None
        
        action_str = "used no potions"
        if use_save and poison_target:
            action_str = f"saved {targeted_player_by_wolves} and poisoned {poison_target}"
        elif use_save:
            action_str = f"saved {targeted_player_by_wolves}"
        elif poison_target:
            action_str = f"poisoned {poison_target}"

        self._write_reflection(
            action=action_str,
            reasoning=log_string,
            directive_id=applied_id,
        )
        self._metric_events.append({
            "phase": "NIGHT",
            "role": "Witch",
            "had_applicable_directive": bool(applicable),
            "directive_id": applied_id,
            "action_summary": action_str,
            # IEI-relevant: Did the witch act on her asymmetric target knowledge?
            "knew_wolf_target": target_display != "None",
            "wolf_target": targeted_player_by_wolves if target_display != "None" else None,
            "used_save_potion": use_save,
            "used_poison_potion": bool(poison_target),
        })

        return {"use_save_potion": use_save, "poison_target": poison_target}, log_string, resp