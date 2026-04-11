from typing import List, Optional, Tuple
from langchain_core.language_models import BaseChatModel
from players.base_player import BasePlayer, Role
from prompt import WITCH_PROMPT_TEMPLATE, WITCH_SAVE_OR_POISON_PROMPT_TEMPLATE


class Witch(BasePlayer):
    """
    The Witch has two one-time use potions:
    - Save Potion: revives the player targeted by the werewolves.
    - Poison Potion: eliminates a player of the Witch's choosing.
    """

    def __init__(
        self, 
        name: str, 
        model: BaseChatModel, 
        role: Role = Role.WITCH,
        is_alive: bool = True,
        system_prompt: str = WITCH_PROMPT_TEMPLATE, 
        personality: str = ""
    ) -> None:
        super().__init__(name=name, role=role, model=model, is_alive=is_alive, system_prompt=system_prompt, personality=personality,)
        self._save_available: bool = True
        self._poison_available: bool = True
    
    def has_save_potion(self) -> bool:
        """Returns True if the save potion is still available."""
        return self._save_available

    def has_poison_potion(self) -> bool:
        """Returns True if the poison potion is still available."""
        return self._poison_available
        
    def has_any_potion(self) -> bool:
        """Returns True if the Witch has at least one potion left."""
        return self.has_save_potion() or self.has_poison_potion()
    
    def save_or_poison(self, targeted_player_by_wolves: Optional[str], alive_players: List[str]) -> Tuple[dict, dict]:
        """
        Night action: choose whether to use the save potion on the wolves' targetand/or the poison potion on any alive player.
        
        Returns a tuple: (actions_dict, raw_response_dict)
        actions_dict format: {"use_save_potion": bool, "poison_target": str | None}
        """
        # Guard: Dead Witch cannot act
        if not self._is_alive:
            return {"use_save_potion": False, "poison_target": None}, {"error": "Witch is dead."}

        # Guard: Need alive players
        if not alive_players:
            return {"use_save_potion": False, "poison_target": None}, {"error": "No alive players provided."}

        # Hide targets if potions are unavailable to prevent LLM hallucination
        target_display = targeted_player_by_wolves if (self._save_available and targeted_player_by_wolves) else "None"
        alive_display = ", ".join(alive_players) if self._poison_available else "None"

        prompt = WITCH_SAVE_OR_POISON_PROMPT_TEMPLATE.format(
            name=self._name,
            save_available=self._save_available,
            poison_available=self._poison_available,
            targeted_player=target_display,
            alive_players=alive_display,
            suspicions=(
                ", ".join(f"{p} ({s:.2f})" for p, s in self._suspicions.items())
                if self._suspicions
                else "No suspicions recorded yet."
            ),
            self_reflections=self._self_reflections or "No specific guidelines yet.",
            notes=", ".join(self._current_game_notes) or "No notes yet.",
        )
        
        resp = self.call_model(prompt, max_tokens=300)
        
        # Extract intents
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
            
        if poison_target and (not self._poison_available or poison_target not in alive_players):
            poison_target = None
            resp["fallback_poison"] = "Forced None: Poison unavailable or target is invalid/dead."
        
        # ---------------------------------------------------------
        # State Update & Logging
        # ---------------------------------------------------------
        
        action_notes = []
        save_reason = resp.get("save_analysis")
        poison_reason = resp.get("poison_analysis")

        if use_save:
            self._save_potion_used = True
            action_notes.append(f"Used SAVE potion on {targeted_player_by_wolves} (Reason: {save_reason}).")
            
        if poison_target:
            self._poison_potion_used = True
            action_notes.append(f"Used POISON potion on {poison_target} (Reason: {poison_reason}).")
            
        if not use_save and not poison_target:
            action_notes.append("Used NO potions tonight.")

        self._add_note(f"Night action: {' '.join(action_notes)}")

        actions = {
            "use_save_potion": use_save,
            "poison_target": poison_target
        }

        return actions, resp
    
if __name__ == "__main__":
    # command to run: /opt/miniconda3/envs/scc452-badass-werewolf/bin/python -m players.witch
    # Dummy data to test the string formatting
    test_prompt = WITCH_SAVE_OR_POISON_PROMPT_TEMPLATE.format(
        name="Wanda",
        save_available=True,
        poison_available=True,
        targeted_player="Bob",
        alive_players="Wanda, Charlie, Dave, Eve",
        suspicions="Charlie (0.85), Dave (0.10)",
        self_reflections="Save confirmed villagers. Poison anyone who fakes a Seer claim.",
        notes="Bob claimed Seer on Day 1. Charlie voted for a confirmed villager."
    )
    
    print("=== WITCH NIGHT ACTION PROMPT ===")
    print(test_prompt)
    print("=================================")