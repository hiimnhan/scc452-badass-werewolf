from langchain_core.language_models import BaseChatModel
from players.base_player import BasePlayer, Role
from prompt import VILLAGER_PROMPT_TEMPLATE


class Villager(BasePlayer):
    """
    Plain Villager — no night action.
    Inherits all day-phase behaviour (debate, vote, bid) from BasePlayer.
    """

    def __init__(
        self,
        name: str,
        model: BaseChatModel,
        game_id: str = "",
        scenario: str = "baseline",
        role: Role = Role.VILLAGER,
        is_alive: bool = True,
        system_prompt: str = VILLAGER_PROMPT_TEMPLATE,
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
