from __future__ import annotations

from collections import defaultdict
from typing import Literal, Optional

import tqdm
import random
from players.base_player import Role
from utils import log_game_summary
from concurrent.futures import ThreadPoolExecutor
from langchain_core.runnables import RunnableConfig
from enum import Enum
import random

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
        self._saved: str = None # the name of the player has been saved
        self._poisoned: str = None # the name of the player has been poisoned
        self._exiled = None
        self._votes = {}
        self._bids = []
        self._debate_log = [] # Log all statements from day discussions. Coach and players will analyze at the end of the game.
        self._summaries = []

        self._vote_logs = [] # Log all votes. Coach and players will analyze at the end of the game.
        self._bid_logs = []
        self._summary_logs = [] # Log all game announcements here for the coach and players to analyze at the end of the game.
        self._protect_log = None
        self._eliminate_log = None
        self._unmask_log = None
        self._witch_log = None

        self._game_logs = []

        self._deception_history = {}
        self._deception_scores = {}
        self._deception_iterations = []
        self._current_speaker = None
        self._winner = None

        self._phase: Phase = Phase.WOLF_DEBATE
        self._step: int = 0

        self._log_dir = None
        self._log_run_id = None
        self._log_paths = {}

    def _compute_current_winner(self, state: GameState) -> Optional[Literal["Villagers", "Werewolves"]]:
        """Compute winner based on current alive players.

        Villagers win if no Werewolves remain.
        Werewolves win if Werewolves >= Villagers.
        Otherwise, no winner yet.
        """
        wolves_alive = [p for p in state._werewolves if p in state._alive_players]
        villagers_alive = [p for p in state._alive_players if p not in wolves_alive]

        if not wolves_alive:
            return "Villagers"
        if len(wolves_alive) >= len(villagers_alive):
            return "Werewolves"
        return None

    def wolf_debate_node(self, state: GameState, config: RunnableConfig) -> GameState:
        player_objects = config.get("configurable", {}).get("player_objects", {})
        active_wolves = [w for w in state._werewolves if w in state._alive_players]
    
        if not active_wolves: # Just in case, if the wolves are actually killed.
            state._phase = Phase.ELIMINATE
            return state

        # 2 rounds of talking, one for proposing and another one for thinking to stick with it or not.
        for _ in range(2): 
            for name in active_wolves:
                wolf_obj = player_objects.get(name)
                others = [w for w in active_wolves if w != name]
                
                statement, log = wolf_obj.wolf_debate(
                    state._alive_players, 
                    others, 
                    state._debate_log
                )
                
                # Save the dialogue to the history so they can refer to it in 'eliminate'
                state._debate_log.append([name, statement])
                
                state._game_logs.append({
                    "phase": "wolf_debate",
                    "player": name,
                    "data": log 
                })

        state._phase = Phase.ELIMINATE
        return state

    def eliminate_node(self, state: GameState, config: RunnableConfig) -> GameState:
        player_objects = config.get("configurable", {}).get("player_objects", {})
        active_wolves = [w for w in state._werewolves if w in state._alive_players]
        
        if not active_wolves: # Just in case, if the wolves are actually killed.
            state._phase = Phase.PROTECT
            return state

        final_votes = {}
        raw_logs = {}

        # Each wolf choose their final player to kill
        for name in active_wolves:
            wolf_obj = player_objects.get(name)
            target, log = wolf_obj.eliminate(state._alive_players)
            final_votes[name] = target
            raw_logs[name] = log # Store the log (analysis, etc.)

        # For tie breaker, to handle situation if there's two names 
        choices = list(set(final_votes.values()))

        if len(choices) == 1:
            chosen_target = choices[0]
        else:
            chosen_target = random.choice(choices) # If there's 2 different name, pick 1 randomly

        # Update state with the final result
        state._eliminated = chosen_target
        state._eliminate_log = str({
            "final_target": chosen_target,
            "votes": final_votes,
            "decisions_metadata": raw_logs
        })
        
        tqdm.tqdm.write(f"Wolves picked {chosen_target}")

        state._phase = Phase.PROTECT
        return state
    
    def protect_node(self, state: GameState, config: RunnableConfig) -> GameState:
        """Guard chooses a player to protect during the night."""
        player_objects = config.get("configurable", {}).get("player_objects", {})
        guard_name = state._guard
        guard_obj = player_objects.get(guard_name)
        # check if guard was killed 
        if guard_name not in state._alive_players:
            state._phase = Phase.UNMASK
            return state

        protect_target, log = guard_obj.protect(state._alive_players)

        if not protect_target:
            raise ValueError(f"{guard_name} failed to specify a protection target.")

        tqdm.tqdm.write(f"{guard_name} protected {protect_target}")

        # Convert log to string if it's a dict
        log_str = str(log) if isinstance(log, dict) else log
        
        state._protected = protect_target
        state._protect_log = log_str
        state._phase = Phase.UNMASK

        #log event haven't been implemented yet.
        # state = log_event(state, "protect", guard_name, {
        # "target": protect_target,
        # "raw_output": log
        # })
        return state

    def unmask_node(self, state: GameState) -> GameState:
        # TODO: UNMASK
        return state

    def save_or_poison_node(self, state: GameState, config: RunnableConfig) -> GameState:
        """Witch decides whether to use her save and/or poison potions."""
        # Retrieve player objects from the LangGraph config
        player_objects = config.get("configurable", {}).get("player_objects", {})
        witch_name = state._witch
        witch_obj = player_objects.get(witch_name)

        # Transition to the next phase if there is no witch or the witch is dead
        if not witch_name or witch_name not in state._alive_players:
            state._phase = Phase.RESOLVE_NIGHT
            return state
        
        # Skip if the witch has already used both potions!
        if not witch_obj.has_any_potion():
            tqdm.tqdm.write(f"{witch_name} has no potions left. Skipping turn.")
            state._saved = None
            state._poisoned = None
            state._witch_log = "Skipped: No potions available."
            state._phase = Phase.RESOLVE_NIGHT
            return state

        # The wolves' target from the ELIMINATE phase is typically stored in state._eliminated
        target_by_wolves = getattr(state, "_eliminated", None)

        # Call the Witch's action method
        actions, log = witch_obj.save_or_poison(
            targeted_player_by_wolves=target_by_wolves,
            alive_players=state._alive_players,
            round_num=state._round_num
        )

        use_save = actions.get("use_save_potion", False)
        poison_target = actions.get("poison_target")

        # Record the actions in the state so resolve_night_node can process them
        state._saved = target_by_wolves if use_save else None
        state._poisoned = poison_target

        # Print the outcomes to the terminal
        if use_save:
            announcement = f"{witch_name} used SAVE potion on {target_by_wolves}"
        if poison_target:
            announcement = f"{witch_name} used POISON potion on {poison_target}"
        if not use_save and not poison_target:
            announcement = f"{witch_name} did not use any potions."
        
        tqdm.tqdm.write(announcement)
        state = log_game_summary(state, announcement)
        
        # Store the raw log for history/debugging
        state._witch_log = str(log) if isinstance(log, dict) else log
        
        # Advance the phase
        state._phase = Phase.RESOLVE_NIGHT

        return state

    def resolve_night_node(self, state: GameState) -> GameState:
        """Apply elimination/protection/save/poison outcome and broadcast announcement."""
        
        killed_players = []

        # Evaluate Wolf Kill
        if state._eliminated and state._eliminated != state._protected and state._eliminated != state._saved:
            killed_players.append(state._eliminated)

        # Evaluate Witch Poison (bypasses guard and save)
        if state._poisoned:
            killed_players.append(state._poisoned)

        # Apply deaths and format announcement
        if killed_players:
            # Use set() to remove duplicates just in case wolves and witch targeted the same person
            killed_players = list(set(killed_players))
            
            # Remove killed players from the alive list
            state._alive_players = [p for p in state._alive_players if p not in killed_players]
            
            # Safely format the announcement string
            verb = 'was' if len(killed_players) == 1 else 'were'
            announcement = f"{' and '.join(killed_players)} {verb} killed during the night."
        else:
            announcement = "No one was killed during the night."

        tqdm.tqdm.write(announcement)
        state = log_game_summary(state, announcement)

        # Advance State and Log the Announcement
        state._phase = Phase.CHECK_WINNER_NIGHT
        
        return state

    def check_winner_night_node(self, state: GameState) -> GameState:
        """Return to day phase or finish game if a faction wins."""
        winner = self._compute_current_winner(state)
        state._phase = Phase.DEBATE if not winner else Phase.SUMMARIZE
        state._step = 0

        return state

    def debate_node(self, state: GameState, config: RunnableConfig) -> GameState:
        player_objects = config.get("configurable", {}).get("player_objects", {})
        MAX_DEBATE_TURNS = config.get("configurable", {}).get("MAX_DEBATE_TURNS", 6)

        last_speaker = state._debate_log[-1][0] if state._debate_log else None
        next_possible_speakers = [p for p in state._alive_players if p != last_speaker]
        bid_logs = []
        bid_dict = {}

        # 1. Run bids in parallel
        with ThreadPoolExecutor(max_workers=len(next_possible_speakers)) as executor:
            futures = {name: executor.submit(player_objects[name]._get_bid) for name in next_possible_speakers}
            for name, future in futures.items():
                bid, raw_output = future.result()
                bid_dict[name] = bid
                bid_logs.append(f"{name} bid {bid} - {raw_output}")

        # 2. Pick the winner
        max_bid_value = max(bid_dict.values())
        top_bidders = [name for name, bid in bid_dict.items() if bid == max_bid_value]
        chosen_speaker = random.choice(top_bidders)
        
        # 3. Generate the statement (fixed to .debate() without underscore)
        statement, log = player_objects[chosen_speaker].debate()
        
        if not statement:
            raise ValueError(f"{chosen_speaker} failed to produce a debate line.")

        tqdm.tqdm.write(f"{chosen_speaker}: {statement}")
 
        # 4. Make the other players "listen" and update suspicions in parallel!
        listeners = [p for p in state._alive_players if p != chosen_speaker]
        with ThreadPoolExecutor(max_workers=len(listeners)) as executor:
            # We submit the suspicion update for all listeners. 
            # We don't strictly need to track the return futures unless we want to log them.
            for name in listeners:
                executor.submit(player_objects[name]._update_suspicion, chosen_speaker, statement)

        # 5. Mutate State Manually
        state._debate_log.append([chosen_speaker, statement])
        state._bid_logs.extend(bid_logs)
        state._current_speaker = chosen_speaker
        state._step += 1
        
        # Advance phase using your Enum
        if state._step >= MAX_DEBATE_TURNS:
            state._phase = Phase.VOTE
        else:
            state._phase = Phase.DEBATE
        
        return state

    def vote_node(self, state: GameState, config: RunnableConfig) -> GameState:
        """All alive players cast a vote simultaneously to exile someone."""
        player_objects = config.get("configurable", {}).get("player_objects", {})
        
        vote_dict = {}
        vote_logs = []

        # 1. Run voting in parallel for all alive players
        with ThreadPoolExecutor(max_workers=len(state._alive_players)) as executor:
            # Notice we pass state._alive_players as an argument to the .vote() method!
            futures = {
                name: executor.submit(player_objects[name].vote, state._alive_players) 
                for name in state._alive_players
            }
            
            for name, future in futures.items():
                target, raw_output = future.result()
                vote_dict[name] = target
                
                # Extract the public reasoning for the game history
                reasoning = raw_output.get("reasoning", "No public reason provided.")
                vote_logs.append(f"{name} voted to exile {target}. Reason: {reasoning}")

        # 2. Tally the votes
        vote_counts = defaultdict(int)
        for target in vote_dict.values():
            if target:  # Failsafe in case a model hallucinates or returns an error
                vote_counts[target] += 1

        # 3. Determine the outcome (and handle ties!)
        if vote_counts:
            max_votes = max(vote_counts.values())
            # Find everyone who tied for the highest number of votes
            tied_players = [p for p, c in vote_counts.items() if c == max_votes]
            
            # Standard Werewolf rule: if there is a tie, pick randomly (or you can code a re-vote mechanic later!)
            exiled_player = random.choice(tied_players)
        else:
            exiled_player = None
            max_votes = 0

        # 4. Print the dramatic results to your terminal
        tqdm.tqdm.write("\n=== VOTING RESULTS ===")
        for log in vote_logs:
            tqdm.tqdm.write(log)
            
        if exiled_player:
            tqdm.tqdm.write(f"\n=> {exiled_player} has the most votes ({max_votes}) and will be exiled!")

        # 5. Mutate State manually
        state._votes = vote_dict
        state._vote_logs.extend(vote_logs)
        state._exiled = exiled_player
        
        # Advance phase
        state._phase = Phase.EXILE
        
        return state

    def exile_node(self, state: GameState, config: RunnableConfig) -> GameState:
        """Apply the vote outcome, broadcast announcement, and flush notes to disk."""
        player_objects = config.get("configurable", {}).get("player_objects", {})
        
        exiled_player = state._exiled
        
        # 1. Apply the elimination and format the announcement
        if exiled_player:
            # Safely remove the player from the alive list
            state._alive_players = [p for p in state._alive_players if p != exiled_player]
            announcement = f"{exiled_player} was exiled by the village."
        else:
            announcement = "The village could not reach a decision, and no one was exiled."
            
        # Print to terminal and save to game history
        tqdm.tqdm.write(f"\n=> {announcement}")
        state = log_game_summary(state, announcement)
        
        # 2. Broadcast the announcement to ALL currently alive players
        for name in state._alive_players:
            if name in player_objects:
                player_objects[name].receive_announcement(
                    round_num=state._round_num,
                    phase="Day",
                    announcement=announcement
                )
                
        # 3. Flush notes to disk (Disk I/O is slow, so we use parallel threading!)
        # We also want the exiled player to write their final note before they "die"
        players_to_compile = state._alive_players.copy()
        if exiled_player and exiled_player not in players_to_compile:
            players_to_compile.append(exiled_player)
            
        with ThreadPoolExecutor(max_workers=len(players_to_compile)) as executor:
            for name in players_to_compile:
                if name in player_objects:
                    executor.submit(player_objects[name]._compile_note)

        # 4. Advance State
        state._phase = Phase.CHECK_WINNER_DAY
        
        return state

    def check_winner_day_node(self, state: GameState) -> GameState:
        winner = self._compute_current_winner(state)
        state._phase = Phase.WOLF_DEBATE if not winner else Phase.SUMMARIZE
        state._step = 0

        return state

    def summarize_node(self, state: GameState) -> GameState:
        # TODO: SUMMARIZE
        return state

    def end_node(self, state: GameState) -> GameState:
        # TODO: END
        return state

    def build_graph(self):
        graph = StateGraph(GameState)

        graph.add_node("wolf_debate", self.wolf_debate_node)
        graph.add_node("eliminate", self.eliminate_node)
        graph.add_node("protect", self.protect_node)
        graph.add_node("unmask", self.unmask_node)
        graph.add_node("save_or_poison", self.save_or_poison_node)
        graph.add_node("resolve_night", self.resolve_night_node)
        graph.add_node("check_winner_night", self.check_winner_night_node)
        graph.add_node("debate", self.debate_node)
        graph.add_node("vote", self.vote_node)
        graph.add_node("exile", self.exile_node)
        graph.add_node("check_winner_day", self.check_winner_day_node)
        graph.add_node("summarize", self.summarize_node)
        graph.add_node("end", self.end_node)

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
