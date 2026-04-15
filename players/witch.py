from typing import List, Optional, Tuple
from langchain_core.language_models import BaseChatModel
from players.base_player import BasePlayer, Role
from prompt import WITCH_PROMPT_TEMPLATE, WITCH_SAVE_OR_POISON_PROMPT_TEMPLATE


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
        role: Role = Role.WITCH,
        is_alive: bool = True,
        system_prompt: str = WITCH_PROMPT_TEMPLATE,
        personality: str = "",
    ) -> None:
        super().__init__(name=name, role=role, model=model, game_id=game_id, is_alive=is_alive, system_prompt=system_prompt, personality=personality)
        self._save_available: bool = True
        self._poison_available: bool = True

    def has_save_potion(self) -> bool:
        return self._save_available

    def has_poison_potion(self) -> bool:
        return self._poison_available

    def has_any_potion(self) -> bool:
        """Returns True if the Witch has at least one potion left."""
        return self.has_save_potion() or self.has_poison_potion()
    
    def save_or_poison(self, targeted_player_by_wolves: Optional[str], alive_players: List[str], round_num: int) -> Tuple[dict, dict]:
        """Night action: decide whether to use the save and/or poison potion.

        Returns (actions_dict, raw_response_dict).
        actions_dict format: {"use_save_potion": bool, "poison_target": str | None}
        """
        if not self._is_alive:
            return {"use_save_potion": False, "poison_target": None}, {"error": "Witch is dead."}

        if not alive_players:
            return {"use_save_potion": False, "poison_target": None}, {"error": "No alive players provided."}

        # The Witch only learns the wolf target while her Save potion is available.
        # Once the Save potion is spent, the wolf target is hidden from her entirely.
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
        
        resp = self.call_model(prompt, max_tokens=300)
        
        # ------------------------------------------------------------------
        # Extract intents
        # ------------------------------------------------------------------
        use_save = resp.get("use_save_potion", False)
        poison_target = resp.get("poison_target", "None")
        
        # ---------------------------------------------------------
        # Validation & Fallbacks
        # ---------------------------------------------------------
        
        # 1. Validate Save Potion
        if use_save and (not self._save_available or not targeted_player_by_wolves):
            use_save = False
            resp["fallback_save"] = "Forced False: Save potion unavailable or no active target."

        # 2. Validate Poison Potion
        if isinstance(poison_target, str) and poison_target.lower() == "none":
            poison_target = None
            
        if poison_target and (not self._poison_available or poison_target not in alive_players_except_witch):
            poison_target = None
            resp["fallback_poison"] = "Forced None: Poison unavailable or target is invalid/dead."

        # ------------------------------------------------------------------
        # State update & own-action logging
        # ------------------------------------------------------------------
        save_reason   = resp.get("save_analysis")
        poison_reason = resp.get("poison_analysis")

        if use_save:
            self._save_available = False
            self._record_own_action(
                round_num, "Night",
                f"Used SAVE potion on {targeted_player_by_wolves}. Reason: {save_reason}."
            )

        if poison_target:
            self._poison_available = False
            self._record_own_action(
                round_num, "Night",
                f"Used POISON potion on {poison_target}. Reason: {poison_reason}."
            )

        if not use_save and not poison_target:
            self._record_own_action(round_num, "Night", "Used no potions tonight.")

        return {"use_save_potion": use_save, "poison_target": poison_target}, resp