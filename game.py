from __future__ import annotations
from typing import TYPE_CHECKING

from collections import defaultdict
from typing import Literal, Optional

import tqdm
from config import SCENARIO_CONFIG
from players.base_player import Role
from utils import log_game_summary
from concurrent.futures import ThreadPoolExecutor
from langchain_core.runnables import RunnableConfig
from enum import Enum
import random
import math

from langgraph.graph import StateGraph, END

if TYPE_CHECKING: # for type checking purposes
    from players.witch import Witch
    from players.guard import Guard
    from players.seer import Seer
    from players.coach import Coach
    from players.wolf import Wolf


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
    # SUMMARIZE = "summarize"
    END = "end"


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
        self._saved: str = "" # the name of the player has been saved
        self._poisoned: str = "" # the name of the player has been poisoned
        self._exiled: str = ""
        self._wolf_debate_log: dict[int, list] = defaultdict(
            list
        ) # Log all night discussions between wolves. {round_num: [list of statements]}
        self._debate_log: dict[int, list] = defaultdict(
            list
        ) # Log all statements from day discussions. Coach will analyze at the end of the game. Players don't use it as they have their own summary to analyze in their _note already.
        self._vote_logs = [] # Log all votes. Coach and players will analyze at the end of the game.
        self._bid_logs = []
        self._summary_logs = [] # Log all game announcements here for the coach to analyze at the end of the game. Players don't use it as they have their own summary to analyze in their _note already.

        self._deception_history = {}
        self._deception_scores = {}
        self._deception_iterations = []
        self._current_speaker = None
        self._winner: Literal["Villagers", "Werewolves"] | None = None

        self._phase: Phase = Phase.WOLF_DEBATE
        self._step: int = 0

    def _compute_current_winner(self, state: GameState) -> Optional[Literal["Villagers", "Werewolves"]]:
        """Compute winner based on current alive players.

        Villagers win if no Werewolves remain.
        Werewolves win if Werewolves >= Villagers.
        Otherwise, no winner yet.
        """
        wolves_alive = [p for p in state._werewolves if p in state._alive_players]
        villagers_alive = [p for p in state._alive_players if p not in wolves_alive]

        if not wolves_alive:
            self._winner = "Villagers"
        elif len(wolves_alive) >= len(villagers_alive):
            self._winner = "Werewolves"

        if self._winner:
            return self._winner

        return None

    def wolf_debate_node(self, state: GameState, config: RunnableConfig) -> GameState:
        player_objects = config.get("configurable", {}).get("player_objects", {})
        active_wolves = [w for w in state._werewolves if w in state._alive_players]

        if (not active_wolves) or (
            len(active_wolves) == 1
        ): # If all the wolves are actually killed or there is only 1 wolf, immediately go to ELIMINATE phase.
            state._phase = Phase.ELIMINATE
            return state

        # 2 rounds of talking, one for proposing and another one for thinking to stick with it or not.
        for _ in range(2):
            for name in active_wolves:
                wolf_obj: Wolf = player_objects.get(name)
                others = [w for w in active_wolves if w != name]

                statement, log = wolf_obj.wolf_debate(
                    state._alive_players,
                    others,
                    state._wolf_debate_log[state._round_num],
                    state._round_num,
                )

                # Save the dialogue to the history so they can refer to it in 'eliminate'
                state._wolf_debate_log[state._round_num].append([name, statement])

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
            wolf_obj: Wolf = player_objects.get(name)
            target, log = wolf_obj.eliminate(state._alive_players, state._round_num)
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

        announcement = f"{chosen_target} is targeted by the wolves."
        tqdm.tqdm.write(announcement)
        state = log_game_summary(state, announcement)

        state._phase = Phase.PROTECT
        return state

    def protect_node(self, state: GameState, config: RunnableConfig) -> GameState:
        """Guard chooses a player to protect during the night."""
        player_objects = config.get("configurable", {}).get("player_objects", {})
        guard_name = state._guard
        guard_obj: Guard = player_objects.get(guard_name)
        # check if guard was killed
        if guard_name not in state._alive_players:
            state._phase = Phase.UNMASK
            return state

        protect_target, log = guard_obj.protect(state._alive_players, state._round_num)

        if not protect_target:
            raise ValueError(f"{guard_name} failed to specify a protection target.")

        announcement = f"{guard_name} protected {protect_target}"
        tqdm.tqdm.write(announcement)
        state = log_game_summary(state, announcement)

        state._protected = protect_target
        state._phase = Phase.UNMASK

        return state

    def unmask_node(self, state: GameState, config: RunnableConfig) -> GameState:
        """Seer secretly investigates one player during the night.

        Moderator check: the seer receives ONLY the binary wolf/not-wolf
        answer — never the exact role. The result is piped into the
        Seer object via reveal_and_update().
        """
        player_objects = config.get("configurable", {}).get("player_objects", {})
        seer_name = state._seer
        seer_obj: Seer = player_objects.get(seer_name)

        # Skip if no seer in game or seer has been eliminated
        if not seer_name or seer_name not in state._alive_players:
            state._phase = Phase.SAVE_OR_POISON
            return state

        # Seer picks a target
        target, log = seer_obj.unmask(state._alive_players, state._round_num)

        # Guard against empty response (error case — e.g. only self alive)
        if not target:
            tqdm.tqdm.write(f"{seer_name} did not investigate this round.")
            state._phase = Phase.SAVE_OR_POISON
            return state

        # Moderator check — seer only learns the binary answer
        is_wolf = target in state._werewolves
        seer_obj.reveal_and_update(target, is_wolf, state._round_num)

        # Private terminal output (for debugging; never sent to players)
        announcement = f"{seer_name} investigated {target} — result: {'WOLF' if is_wolf else 'not a wolf'}"
        tqdm.tqdm.write(announcement)

        # Game-level summary log (visible to coach post-game, not to players)
        state = log_game_summary(state, announcement)

        # Store moderator-side record
        state._unmasked = target

        # Advance phase
        state._phase = Phase.SAVE_OR_POISON
        return state

    def save_or_poison_node(self, state: GameState, config: RunnableConfig) -> GameState:
        """Witch decides whether to use her save and/or poison potions."""
        # Retrieve player objects from the LangGraph config
        player_objects = config.get("configurable", {}).get("player_objects", {})
        witch_name = state._witch
        witch_obj: Witch = player_objects.get(witch_name)

        # Transition to the next phase if there is no witch or the witch is dead
        if not witch_name or witch_name not in state._alive_players:
            state._phase = Phase.RESOLVE_NIGHT
            return state

        # Skip if the witch has already used both potions!
        if not witch_obj.has_any_potion():
            tqdm.tqdm.write(f"{witch_name} has no potions left. Skipping turn.")
            state._saved = ""
            state._poisoned = ""
            state._phase = Phase.RESOLVE_NIGHT
            return state

        # The wolves' target from the ELIMINATE phase is typically stored in state._eliminated
        target_by_wolves = getattr(state, "_eliminated", None)

        # Call the Witch's action method
        actions, log = witch_obj.save_or_poison(
            targeted_player_by_wolves=target_by_wolves,
            alive_players=state._alive_players,
            round_num=state._round_num,
        )

        use_save = actions.get("use_save_potion", False)
        poison_target = actions.get("poison_target")

        # Record the actions in the state so resolve_night_node can process them
        state._saved = str(target_by_wolves) if (use_save and target_by_wolves) else ""
        state._poisoned = str(poison_target) if poison_target else ""

        # Print the outcomes to the terminal
        action_parts = []
        if use_save:
            action_parts.append(f"used SAVE potion on {target_by_wolves}")
        if poison_target:
            action_parts.append(f"used POISON potion on {poison_target}")
        if action_parts:
            announcement = f"{witch_name} {' and '.join(action_parts)}"
        else:
            announcement = f"{witch_name} did not use any potions."

        tqdm.tqdm.write(announcement)
        state = log_game_summary(state, announcement)

        # Advance the phase
        state._phase = Phase.RESOLVE_NIGHT

        return state

    def resolve_night_node(self, state: GameState, config: RunnableConfig) -> GameState:
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
            verb = "was" if len(killed_players) == 1 else "were"
            announcement = f"{' and '.join(killed_players)} {verb} killed during the night."
        else:
            announcement = "No one was killed during the night."

        tqdm.tqdm.write(announcement)
        state = log_game_summary(state, announcement)

        # Broadcast to ALL alive players so their _note stays current
        player_objects = config.get("configurable", {}).get("player_objects", {})
        with ThreadPoolExecutor(max_workers=max(1, len(state._alive_players))) as executor:
            threads = [
                executor.submit(
                    player_objects[name].receive_announcement,
                    state._round_num,
                    "Night",
                    announcement,
                )
                for name in state._alive_players
                if name in player_objects
            ]
            for thread in threads:
                thread.result()

        # Advance State
        state._phase = Phase.CHECK_WINNER_NIGHT

        return state

    def check_winner_night_node(self, state: GameState) -> GameState:
        """Return to day phase or finish game if a faction wins."""
        winner = self._compute_current_winner(state)
        state._phase = Phase.DEBATE if not winner else Phase.END
        state._step = 0

        return state

    def debate_node(self, state: GameState, config: RunnableConfig) -> GameState:
        player_objects = config.get("configurable", {}).get("player_objects", {})
        MAX_DEBATE_TURNS = config.get("configurable", {}).get("MAX_DEBATE_TURNS", 6)

        current_round_log = state._debate_log.get(state._round_num, [])
        last_speaker = current_round_log[-1][0] if current_round_log else None

        next_possible_speakers = [p for p in state._alive_players if p != last_speaker]
        bid_logs = []
        bid_dict = {}

        # 1. Run bids in parallel
        with ThreadPoolExecutor(max_workers=len(next_possible_speakers)) as executor:
            futures = {name: executor.submit(player_objects[name].get_bid) for name in next_possible_speakers}
            for name, future in futures.items():
                bid, raw_output = future.result()
                bid_dict[name] = bid
                bid_logs.append(f"{name} bid {bid} - {raw_output}")

        # 2. Pick the winner
        max_bid_value = max(bid_dict.values())
        top_bidders = [name for name, bid in bid_dict.items() if bid == max_bid_value]
        chosen_speaker = random.choice(top_bidders)

        # 3. Generate the statement (with a safety net)
        statement = None
        retries = 0
        while not statement and retries < 3:
            statement, log = player_objects[chosen_speaker].debate()
            retries += 1

        # Fallback if the LLM completely fails
        if not statement:
            statement = "I have nothing to add at this moment."

        tqdm.tqdm.write(f"{chosen_speaker}: {statement}")

        # 4. Make the other players "listen" and update suspicions in parallel!
        listeners = [p for p in state._alive_players if p != chosen_speaker]
        with ThreadPoolExecutor(max_workers=len(listeners)) as executor:
            threads = [
                executor.submit(
                    player_objects[name].update_suspicion_from_statement,
                    chosen_speaker,
                    statement,
                )
                for name in listeners
            ]

            # Pause the game until everyone finishes updating their notes!
            for thread in threads:
                thread.result()

        # 5. Mutate State Manually
        state._debate_log[state._round_num].append((chosen_speaker, statement))
        state._bid_logs.extend(bid_logs)
        state._current_speaker = chosen_speaker
        state._step += 1

        # 6. Advance phase
        if state._step >= MAX_DEBATE_TURNS:
            state._phase = Phase.VOTE
            state._step = 0
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
            futures = {
                name: executor.submit(player_objects[name].vote, state._alive_players) for name in state._alive_players
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
            if target:
                vote_counts[target] += 1

        # 3. Determine the outcome
        threshold = math.ceil(len(state._alive_players) / 2.0)

        if vote_counts:
            max_votes = max(vote_counts.values())
            # Find everyone who tied for the highest number of votes
            tied_players = [p for p, c in vote_counts.items() if c == max_votes]

            if max_votes >= threshold:
                exiled_player = random.choice(tied_players)
            else:
                exiled_player = ""
        else:
            exiled_player = ""
            max_votes = 0
            tied_players = []

        # 4. Print the dramatic results to your terminal
        tqdm.tqdm.write("\n=== VOTING RESULTS ===")
        for log in vote_logs:
            tqdm.tqdm.write(log)

        if exiled_player:
            tqdm.tqdm.write(f"\n=> {exiled_player} received {max_votes} votes. They will be exiled!")
        else:
            if max_votes > 0:
                tqdm.tqdm.write(
                    f"\n=> The highest vote count was {max_votes} for {' and '.join(tied_players)} (Require at least {threshold} vote{'s' if threshold > 1 else ''}). Not enough consensus. No one is exiled."
                )
            else:
                tqdm.tqdm.write("\n=> No valid votes were cast. No one is exiled.")

        # 5. Mutate State manually
        state._vote_logs.append(vote_logs)
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
        with ThreadPoolExecutor(max_workers=len(state._alive_players)) as executor:
            threads = [
                executor.submit(
                    player_objects[name].receive_announcement,
                    state._round_num,
                    "Day",
                    announcement,
                )
                for name in state._alive_players
                if name in player_objects
            ]
            # Wait for all players to receive the message
            for thread in threads:
                thread.result()

        # 3. We want the exiled player to write their final note before they "die"
        if exiled_player and exiled_player in player_objects:
            player_objects[exiled_player].compile_note()

        # 4. Advance State
        state._phase = Phase.CHECK_WINNER_DAY

        return state

    def check_winner_day_node(self, state: GameState, config: RunnableConfig) -> GameState:
        player_objects = config.get("configurable", {}).get("player_objects", {})

        winner = self._compute_current_winner(state)
        if not winner:
            state._phase = Phase.WOLF_DEBATE
            state._round_num += 1
            announcement = "After voting, the game continues."
        else:
            state._phase = Phase.END
            announcement = "After voting, the game ends."

        analyzers = state._alive_players.copy()
        if state._exiled and state._exiled not in analyzers:
            analyzers.append(state._exiled)

        current_votes = state._vote_logs[-1] if state._vote_logs else []

        with ThreadPoolExecutor(max_workers=len(analyzers)) as executor:
            threads = [
                executor.submit(
                    player_objects[name].update_suspicion_from_vote,
                    current_votes,
                    state._exiled,
                    announcement,
                )
                for name in analyzers
                if name in player_objects
            ]

            # Pause the game right here until every single thread finishes its work!
            for thread in threads:
                thread.result() # We don't save the result, we just wait for it to finish.

        state._step = 0
        state._round_num += 1

        return state

    def end_node(self, state: GameState, config: RunnableConfig) -> GameState:
        player_objects = config.get("configurable", {}).get("player_objects", {})
        coach_object: Coach | None = config.get("configurable", {}).get("coach", None) # Passed in from your run.py!

        scenario = config.get("configurable", {}).get("scenario", "baseline")
        scenario_config = SCENARIO_CONFIG[scenario]

        tqdm.tqdm.write(f"\n=== GAME OVER ===\nWinner: {state._winner}")

        # 1. Update surviving players' notes
        # (The dead players already compiled their notes when they died!)
        with ThreadPoolExecutor(max_workers=len(state._alive_players)) as executor:
            threads = [
                executor.submit(player_objects[name].compile_note)
                for name in state._alive_players
                if name in player_objects
            ]
            # Pause the game until all files are safely written to disk
            for thread in threads:
                thread.result()

        # 2. Generate coach's feedback and update coach strategy
        if scenario_config.get("coaching"):
            if coach_object:
                tqdm.tqdm.write("=> Coach is reviewing the game and writing feedback...")
                # The coach needs the full public game record to analyze what happened
                game_record = "\n".join(state._summary_logs) if state._summary_logs else "No events recorded."
                coach_object.run(game_record)
            else:
                tqdm.tqdm.write("=> WARNING: Coaching is enabled, but no coach object was provided in config.")

        # 3. Update and write out strategy for ALL players (alive and dead)
        # We run this in parallel so you don't have to wait minutes for the game to close!
        tqdm.tqdm.write("=> Players are analyzing their performance and updating strategies...")
        with ThreadPoolExecutor(max_workers=len(player_objects)) as executor:
            threads = [
                executor.submit(
                    player.update_strategy,
                    # self_analyze=scenario_config.get("self_analyze", False),
                    # coaching=scenario_config.get("coaching", False),
                )
                for player in player_objects.values()
            ]
            # Wait for everyone to finish writing to disk
            for thread in threads:
                thread.result()

        # 4. Write per-game log files to disk
        from pathlib import Path
        from utils import write_to_file

        game_dir = (
            Path(__file__).parent
            / "game_logs"
            / scenario
            / f"game_{config.get('configurable', {}).get('game_id', 'unknown')}"
        ).resolve()

        # game_summary.txt — every public announcement in order
        write_to_file(
            game_dir / "game_summary.txt",
            "\n".join(state._summary_logs) if state._summary_logs else "No events recorded.",
        )

        # debate_log.txt — all statements from all day rounds
        debate_lines = []
        for round_num in sorted(state._debate_log):
            for speaker, text in state._debate_log[round_num]:
                debate_lines.append(f"Round {round_num} | {speaker}: {text}")
        write_to_file(
            game_dir / "debate_log.txt",
            "\n".join(debate_lines) if debate_lines else "No debate recorded.",
        )

        # vote_log.txt — all vote entries across all rounds
        vote_lines = []
        for round_votes in state._vote_logs:
            vote_lines.extend(round_votes)
        write_to_file(
            game_dir / "vote_log.txt",
            "\n".join(vote_lines) if vote_lines else "No votes recorded.",
        )

        # roles_this_game.txt — who played what role (useful for analysis)
        role_lines = [f"{name}: {role.value}" for name, role in state._roles.items()]
        write_to_file(game_dir / "roles_this_game.txt", "\n".join(role_lines))

        tqdm.tqdm.write("=> Game successfully wrapped up. Ready for the next round!")
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
        # graph.add_node("summarize", self.summarize_node)
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
        """ graph.add_conditional_edges("summarize", lambda s: s._phase) """
        graph.add_edge("end", END)

        return graph.compile()
