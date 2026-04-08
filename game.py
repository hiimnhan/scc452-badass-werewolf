from __future__ import annotations

from collections import defaultdict
from typing import Literal, Optional

import tqdm
from players.base_player import Role
from langchain_core.runnables import RunnableConfig
from enum import Enum

from langgraph.graph import StateGraph, END


class Phase(Enum):
    WOLF_DEBATE = "wolf_debate"
    ELIMINATE = "eliminate"
    PROTECT = "protect"
    UNMASK = "unmask"
    SAVE_OR_POISON = "save_or_poison"
    RESOLVE_NIGHT = "resolve_night"
    CHECK_WINNER_NIGHT = "check_winner_night"
    DEBATE = "debate"
    VOTE = "vote"
    EXILE = "exile"
    CHECK_WINNER_DAY = "check_winner_day"
    SUMMARIZE = "summarize"


class GameState:
    def __init__(
        self,
        round_num: int = 0,
        players: list[str] = [],
        alive_players: list[str] = [],
        villagers: list[str] = [],
        werewolves: list[str] = [],
        seer: str | None = None,
        guard: str | None = None,
        witch: str | None = None,
        roles: dict[str, Role] = defaultdict(),
    ) -> None:

        self._round_num = round_num
        self._players = players
        self._alive_players = alive_players
        self._villagers = villagers
        self._werewolves = werewolves
        self._seer = seer
        self._guard = guard
        self._witch = witch
        self._roles = roles

        self._eliminated = None
        self._protected = None
        self._unmasked = None
        self._exiled = None
        self._votes = {}
        self._bids = []
        self._debate_log = []
        self._summaries = []

        self._vote_logs = []
        self._bid_logs = []
        self._summary_logs = []
        self._protect_log = None
        self._eliminate_log = None
        self._unmask_log = None

        self._game_logs = []

        self._deception_history = {}
        self._deception_scores = {}
        self._deception_iterations = []
        self._current_speaker = None
        self._winner = None

        self._phase: Phase = Phase.WOLF_DEBATE
        self._step = 0

        self._log_dir = None
        self._log_run_id = None
        self._log_paths = {}

    def _compute_current_winner(
        self, state: GameState
    ) -> Optional[Literal["Villagers", "Werewolves"]]:
        # TODO: compute current winner
        return None

    def wolf_debate_node(self, state: GameState) -> GameState:
        # TODO: WOLF_DEBATE
        return state

    def eliminate_node(self, state: GameState) -> GameState:
        # TODO: ELIMINATE
        return state
    
    def protect_node(self, state: GameState, config: RunnableConfig) -> GameState:
        """Guard chooses a player to protect during the night."""
        player_objects = config.get("configurable", {}).get("player_objects", {})
        guard_name = state._guard
        guard_obj = player_objects.get(guard_name)
        # check if guard was killed 
        if guard_name not in state.alive_players:
            return state.model_copy(update={"phase": "unmask"})

        protect_target, log = guard_obj.protect(self._alive_players)

        if not protect_target:
            raise ValueError(f"{guard_name} failed to specify a protection target.")

        tqdm.tqdm.write(f"{guard_name} protected {protect_target}")

        # Convert log to string if it's a dict
        log_str = str(log) if isinstance(log, dict) else log
        
        state =  state.model_copy(update={
            "protected": protect_target,
            "protect_log": log_str,
            "phase": "unmask"
        })

        #log event haven't been implemented yet.
        # state = log_event(state, "protect", guard_name, {
        # "target": protect_target,
        # "raw_output": log
        # })
        return state

    def unmask_node(self, state: GameState) -> GameState:
        # TODO: UNMASK
        return state

    def save_or_poison_node(self, state: GameState) -> GameState:
        # TODO: SAVE_OR_POISON
        return state

    def resolve_night_node(self, state: GameState) -> GameState:
        # TODO: RESOLVE_NIGHT
        return state

    def check_winner_night_node(self, state: GameState) -> GameState:
        # TODO: CHECK_WINNER_NIGHT
        return state

    def debate_node(self, state: GameState) -> GameState:
        # TODO: DEBATE
        return state

    def vote_node(self, state: GameState) -> GameState:
        # TODO: VOTE
        return state

    def exile_node(self, state: GameState) -> GameState:
        # TODO: EXILE
        return state

    def check_winner_day_node(self, state: GameState) -> GameState:
        # TODO: CHECK_WINNER_DAY
        return state

    def summarize_node(self, state: GameState) -> GameState:
        # TODO: SUMMARIZE
        return state

    def end_node(self, state: GameState) -> GameState:
        # TODO: END
        return state

    @classmethod
    def build_graph(cls):
        graph = StateGraph(cls)

        graph.add_node("wolf_debate", cls.wolf_debate_node)
        graph.add_node("eliminate", cls.eliminate_node)
        graph.add_node("protect", cls.protect_node)
        graph.add_node("unmask", cls.unmask_node)
        graph.add_node("save_or_poison", cls.save_or_poison_node)
        graph.add_node("resolve_night", cls.resolve_night_node)
        graph.add_node("check_winner_night", cls.check_winner_night_node)
        graph.add_node("debate", cls.debate_node)
        graph.add_node("vote", cls.vote_node)
        graph.add_node("exile", cls.exile_node)
        graph.add_node("check_winner_day", cls.check_winner_day_node)
        graph.add_node("summarize", cls.summarize_node)
        graph.add_node("end", cls.end_node)

        graph.set_entry_point("wolf_debate")

        graph.add_conditional_edges("wolf_debate", lambda s: s._phase)
        graph.add_conditional_edges("eliminate", lambda s: s._phase)
        graph.add_conditional_edges("protect", lambda s: s._phase)
        graph.add_conditional_edges("unmask", lambda s: s._phase)
        graph.add_conditional_edges("save_or_poison", lambda s: s._phase)
        graph.add_conditional_edges("resolve_night", lambda s: s._phase)
        graph.add_conditional_edges("check_winner_night", lambda s: s._phase)
        graph.add_conditional_edges("debate", lambda s: s._phase)
        graph.add_conditional_edges("vote", lambda s: s._phase)
        graph.add_conditional_edges("exile", lambda s: s._phase)
        graph.add_conditional_edges("check_winner_day", lambda s: s._phase)
        graph.add_conditional_edges("summarize", lambda s: s._phase)
        graph.add_edge("end", END)

        return graph.compile()
