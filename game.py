from __future__ import annotations
from typing import TYPE_CHECKING, Optional, TypedDict

from collections import defaultdict
from typing import Literal, Optional

import tqdm
from config import SCENARIO_CONFIG
from players.base_player import Role
from concurrent.futures import ThreadPoolExecutor
from langchain_core.runnables import RunnableConfig
from enum import Enum
import random
import math
import json
from constants import (
    GAME_SUMMARY_FILENAME,
    WOLF_DEBATE_LOG_FILENAME,
    WOLF_TARGET_LOG_FILENAME,
    GUARD_LOG_FILENAME,
    SEER_LOG_FILENAME,
    WITCH_LOG_FILENAME,
    SUSPICION_LOG_FILENAME,
    PLAYER_NOTE_FILENAME,
)
from pathlib import Path
from utils import write_to_file
import copy

from langgraph.graph import StateGraph, END

if TYPE_CHECKING:  # for type checking purposes
    from players.witch import Witch
    from players.guard import Guard
    from players.seer import Seer
    from players.coach import Coach
    from players.wolf import Wolf
    from players.base_player import BasePlayer


class Phase(Enum):
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
    END = "end"


class GameState:
    def __init__(
        self,
        round_num: int = 1,
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
        self._saved: str | None = None  # the name of the player has been saved
        self._poisoned: str | None = None  # the name of the player has been poisoned
        self._exiled: str | None = None

        self._wolf_debate_log: list[list] = []  # Log all night discussions between wolves.
        self._wolf_target_log: list[list] = []  # Wolves' target for each night, NOT players eliminated each night
        self._guard_log: list[list] = []
        self._seer_log: list[list] = []
        self._witch_log: list[list] = []
        self._suspicion_log: dict[int, dict] = defaultdict(dict)
        self._game_summary_log: dict[
            int, dict
        ] = {}  # Log all public game announcements here for the coach to analyze at the end of the game. Players don't use it as they have their own summary to analyze in their _note already.

        self._winner: Literal["Villagers", "Werewolves"] | None = None

        self._phase: Phase = Phase.ELIMINATE
        self._step: int = 0

    def _compute_current_winner(self, gs: GameState) -> Optional[Literal["Villagers", "Werewolves"]]:
        """Compute winner based on current alive players.

        Villagers win if no Werewolves remain.
        Werewolves win if Werewolves >= Villagers.
        Otherwise, no winner yet.
        """
        wolves_alive = [p for p in gs._werewolves if p in gs._alive_players]
        villagers_alive = [p for p in gs._alive_players if p not in wolves_alive]

        if not wolves_alive:
            gs._winner = "Villagers"
        elif len(wolves_alive) >= len(villagers_alive):
            gs._winner = "Werewolves"

        if gs._winner:
            gs._game_summary_log["winner"] = gs._winner
            return gs._winner

        return None

    def eliminate_node(self, state: GraphState, config: RunnableConfig) -> GraphState:
        gs = state["game"]
        player_objects: dict[str, BasePlayer] = config.get("configurable", {}).get("player_objects", {})

        # Logging suspicion
        round_log = gs._suspicion_log.setdefault(gs._round_num, {})
        for name in gs._alive_players:
            round_log[name] = copy.deepcopy(player_objects[name]._suspicion)

        tqdm.tqdm.write(f"\n\n=== Round {gs._round_num} ===")
        tqdm.tqdm.write("*** Night ***")
        active_wolves = [w for w in gs._werewolves if w in gs._alive_players]
        gs._eliminated = None  # Crucial: Reset for the logic of wolf agreement of target

        if not active_wolves:  # If all the wolves are actually killed
            gs._phase = Phase.PROTECT
            gs._step = 0
            return {"game": gs}

        wolf_targets = []
        for _ in range(3):
            agreed = False  # Used to safely break both loops

            # Let the wolves debate
            for name in active_wolves:
                try:
                    dialogue_history = [statement for statement in gs._wolf_debate_log if statement[0] == gs._round_num]
                except Exception:
                    dialogue_history = []

                wolf_obj: Wolf = player_objects.get(name)

                target, statement, analysis, _ = wolf_obj.eliminate(
                    alive_players=gs._alive_players,
                    wolf_teammates=active_wolves,
                    dialogue_history=dialogue_history,
                    round_num=gs._round_num,
                )

                # Save the dialogue to the history so they can refer to it next time
                gs._wolf_debate_log.append([gs._round_num, name, statement, target, analysis])  # Log _wolf_debate_log
                wolf_targets.append(target)

                # Check for agreement
                if (len(active_wolves) == 1) or ((len(wolf_targets) >= 2) and (wolf_targets[-2] == wolf_targets[-1])):
                    gs._eliminated = target
                    agreed = True
                    break

            if agreed:
                break  # Safely exit the while loop if they agreed!

        if not gs._eliminated:
            # Fallback tiebreaker
            if len(wolf_targets) >= 2:
                gs._eliminated = random.choice([wolf_targets[-2], wolf_targets[-1]])
            elif wolf_targets:
                gs._eliminated = wolf_targets[-1]

        # Update note for each wolf
        for name in active_wolves:
            wolf_obj: Wolf = player_objects.get(name)
            wolf_obj.record_own_action(gs._round_num, "Night", f"Final target: {gs._eliminated}")

        gs._wolf_target_log.append([gs._round_num, gs._eliminated])  # Log _wolf_target_log

        if len(active_wolves) >= 2:
            wolf_action = f"{' and '.join(active_wolves)} (Wolves) targeted {gs._eliminated}."
        else:
            wolf_action = f"{active_wolves[0]} (Wolf) targeted {gs._eliminated}."

        round_log = gs._game_summary_log.setdefault(gs._round_num, {})
        night_actions: list = round_log.setdefault("night_actions", [])
        night_actions.append(wolf_action)

        tqdm.tqdm.write(f"\n• {wolf_action}")

        gs._phase = Phase.PROTECT
        gs._step = 0
        return {"game": gs}

    def protect_node(self, state: GraphState, config: RunnableConfig) -> GraphState:
        """Guard chooses a player to protect during the night."""
        gs = state["game"]
        player_objects = config.get("configurable", {}).get("player_objects", {})
        guard_name = gs._guard
        guard_obj: Guard = player_objects.get(guard_name)
        gs._protected = None

        # check if guard was killed
        if guard_name not in gs._alive_players:
            gs._phase = Phase.UNMASK
            return {"game": gs}

        protect_target, analysis, _ = guard_obj.protect(gs._alive_players, gs._round_num)

        if not protect_target:
            raise ValueError(f"{guard_name} (Guard) failed to specify a protection target.")

        guard_action = f"{guard_name} (Guard) chose to protect {protect_target}."
        tqdm.tqdm.write(f"• {guard_action}")
        gs._guard_log.append([gs._round_num, protect_target, analysis])

        round_log = gs._game_summary_log.setdefault(gs._round_num, {})
        night_actions: list = round_log.setdefault("night_actions", [])
        night_actions.append(guard_action)

        gs._protected = protect_target
        gs._phase = Phase.UNMASK

        return {"game": gs}

    def unmask_node(self, state: GraphState, config: RunnableConfig) -> GraphState:
        """Seer secretly investigates one player during the night.

        Moderator check: the seer receives ONLY the binary wolf/not-wolf
        answer — never the exact role. The result is piped into the
        Seer object via reveal_and_update().
        """
        gs = state["game"]
        player_objects = config.get("configurable", {}).get("player_objects", {})
        seer_name = gs._seer
        seer_obj: Seer = player_objects.get(seer_name)

        # Skip if no seer in game or seer has been eliminated
        if not seer_name or seer_name not in gs._alive_players:
            gs._phase = Phase.SAVE_OR_POISON
            return {"game": gs}

        # Seer picks a target
        target, analysis, _ = seer_obj.unmask(gs._alive_players, gs._round_num)

        # Guard against empty response (error case — e.g. only self alive)
        if not target:
            tqdm.tqdm.write(f"{seer_name} (Seer) did not investigate this round.")
            gs._phase = Phase.SAVE_OR_POISON
            return {"game": gs}

        # Moderator check — seer only learns the binary answer
        is_wolf = target in gs._werewolves
        seer_obj.reveal_and_update(target, is_wolf, gs._round_num)

        # Private terminal output (for debugging; never sent to players)
        seer_action = f"{seer_name} (Seer) investigated {target} — Result: {'WOLF' if is_wolf else 'not a wolf'}."
        tqdm.tqdm.write(f"• {seer_action}")

        # Game-level summary log (visible to coach post-game, not to players)
        gs._seer_log.append([gs._round_num, target, is_wolf, analysis])

        round_log = gs._game_summary_log.setdefault(gs._round_num, {})
        night_actions: list = round_log.setdefault("night_actions", [])
        night_actions.append(seer_action)

        # Store moderator-side record
        gs._unmasked = target

        # Advance phase
        gs._phase = Phase.SAVE_OR_POISON
        return {"game": gs}

    def save_or_poison_node(self, state: GraphState, config: RunnableConfig) -> GraphState:
        """Witch decides whether to use her save and/or poison potions."""
        gs = state["game"]
        # Retrieve player objects from the LangGraph config
        player_objects = config.get("configurable", {}).get("player_objects", {})
        witch_name = gs._witch
        witch_obj: Witch = player_objects.get(witch_name)

        # Careful setting for every time this node runs
        gs._saved = None
        gs._poisoned = None

        # Transition to the next phase if there is no witch or the witch is dead
        if not witch_name or witch_name not in gs._alive_players:
            gs._phase = Phase.RESOLVE_NIGHT
            return {"game": gs}

        round_log = gs._game_summary_log.setdefault(gs._round_num, {})
        night_actions: list = round_log.setdefault("night_actions", [])

        # Skip if the witch has already used both potions!
        if not witch_obj.has_any_potion():
            witch_action = f"{witch_name} (Witch) has no potions left."
            tqdm.tqdm.write(f"• {witch_action}")
            witch_obj.record_own_action(gs._round_num, "Night", "No potions left to perform actions.")
            gs._saved = None
            gs._poisoned = None
            gs._phase = Phase.RESOLVE_NIGHT

            night_actions.append(witch_action)
            return {"game": gs}

        # Call the Witch's action method
        actions, analysis, _ = witch_obj.save_or_poison(
            targeted_player_by_wolves=gs._eliminated,
            alive_players=gs._alive_players,
            round_num=gs._round_num,
        )

        use_save = actions.get("use_save_potion", False)
        poison_target = actions.get("poison_target", "")

        # Record the actions in the state so resolve_night_node can process them
        gs._saved = str(gs._eliminated) if (use_save and gs._eliminated) else None
        gs._poisoned = str(poison_target) if poison_target else None

        # Print the outcomes to the terminal
        action_parts = []
        if use_save:
            action_parts.append(f"used SAVE potion on {gs._eliminated}")
        if poison_target:
            action_parts.append(f"used POISON potion on {poison_target}")
        if action_parts:
            announcement = f"{witch_name} (Witch) {' and '.join(action_parts)}"
        else:
            announcement = f"{witch_name} (Witch) did not use any potions."

        tqdm.tqdm.write(f"• {announcement}")
        gs._witch_log.append([gs._round_num, gs._saved, gs._poisoned, analysis])
        night_actions.append(announcement)

        # Advance the phase
        gs._phase = Phase.RESOLVE_NIGHT

        return {"game": gs}

    def resolve_night_node(self, state: GraphState, config: RunnableConfig) -> GraphState:
        """Apply elimination/protection/save/poison outcome and broadcast announcement."""
        gs = state["game"]

        killed_players = []

        # Evaluate Wolf Kill
        if gs._eliminated and gs._eliminated != gs._protected and gs._eliminated != gs._saved:
            killed_players.append(gs._eliminated)

        # Evaluate Witch Poison (bypasses guard and save)
        if gs._poisoned:
            killed_players.append(gs._poisoned)

        # Apply deaths and format announcement
        if killed_players:
            # Use set() to remove duplicates just in case wolves and witch targeted the same person
            killed_players = list(set(killed_players))

            # Remove killed players from the alive list
            gs._alive_players = [p for p in gs._alive_players if p not in killed_players]

            # Safely format the announcement string
            verb = "was" if len(killed_players) == 1 else "were"
            announcement = f"=> {' and '.join(killed_players)} {verb} killed during the night."
        else:
            announcement = "=> No one was killed during the night."

        # Write announcement to terminal
        tqdm.tqdm.write(announcement)

        # Log dead players after night into _game_summary_log
        round_log = gs._game_summary_log.setdefault(gs._round_num, {})
        round_log["night_eliminated"] = killed_players

        player_objects: dict[str, BasePlayer] = config.get("configurable", {}).get("player_objects", {})

        # 1. Broadcast to ALL players (including the freshly dead, so they know how they died)
        all_to_notify = gs._alive_players + killed_players
        for name in all_to_notify:
            if name in player_objects:
                player_objects[name].receive_announcement(gs._round_num, "Night", announcement)

        # 2. Have ALIVE players update their suspicions in parallel
        if gs._alive_players:
            # tqdm.tqdm.write("\n### Players are analyzing the night's outcome...")
            with ThreadPoolExecutor(max_workers=max(1, len(gs._alive_players))) as executor:
                threads = [
                    executor.submit(player_objects[name].update_suspicion_after_night)
                    for name in gs._alive_players
                    if name in player_objects
                ]

                # Pause the game until everyone is done thinking
                for thread in threads:
                    thread.result()

        # Advance State
        gs._phase = Phase.CHECK_WINNER_NIGHT

        return {"game": gs}

    def check_winner_night_node(self, state: GraphState) -> GraphState:
        """Return to day phase or finish game if a faction wins."""
        gs = state["game"]
        winner = self._compute_current_winner(gs)

        if not winner:
            gs._phase = Phase.DEBATE
            tqdm.tqdm.write("\n*** Day ***")
        else:
            gs._phase = Phase.END

        gs._step = 0

        return {"game": gs}

    def debate_node(self, state: GraphState, config: RunnableConfig) -> GraphState:
        gs = state["game"]
        player_objects: dict[str, BasePlayer] = config.get("configurable", {}).get("player_objects", {})
        MAX_DEBATE_TURNS = config.get("configurable", {}).get("MAX_DEBATE_TURNS", 6)

        # Ensure the round dictionary exists
        round_log = gs._game_summary_log.setdefault(gs._round_num, {})
        # Ensure the day_debate list exists
        debate_log: list[dict] = round_log.setdefault("day_debate", [])

        # Safely get the last speaker
        last_speaker = debate_log[-1]["speaker"] if debate_log else None

        next_possible_speakers = [p for p in gs._alive_players if p != last_speaker]
        bid_dict = {}

        # 1. Run bids in parallel
        with ThreadPoolExecutor(max_workers=len(next_possible_speakers)) as executor:
            futures = {
                name: executor.submit(player_objects[name].get_bid, debate_log, gs._round_num)
                for name in next_possible_speakers
            }
            for name, future in futures.items():
                bid, raw_output = future.result()
                bid_dict[name] = bid

        # 2. Pick the winner
        max_bid_value = max(bid_dict.values())
        top_bidders = [name for name, bid in bid_dict.items() if bid == max_bid_value]
        chosen_speaker = random.choice(top_bidders)

        # 3. Generate the statement (with a safety net)
        statement = None
        retries = 0
        while not statement and retries < 3:
            statement, log = player_objects[chosen_speaker].debate(debate_log, gs._alive_players, gs._round_num)
            retries += 1

        # Fallback if the LLM completely fails
        if not statement:
            statement = "I have nothing to add at this moment."

        tqdm.tqdm.write(f"• {chosen_speaker}: {statement}")

        # Log the statement from the player
        debate_log.append({"speaker": chosen_speaker, "statement": statement})

        # 4. Make the other players "listen" and update suspicions in parallel!
        listeners = [p for p in gs._alive_players if p != chosen_speaker]
        with ThreadPoolExecutor(max_workers=len(listeners)) as executor:
            threads = [
                executor.submit(
                    player_objects[name].update_suspicion_from_statement,
                    debate_log,
                    chosen_speaker,
                    gs._round_num,
                )
                for name in listeners
            ]

            # Pause the game until everyone finishes updating their notes!
            for thread in threads:
                thread.result()

        # 5. Mutate State Manually
        gs._step += 1

        # 6. Advance phase
        if gs._step >= MAX_DEBATE_TURNS:
            gs._phase = Phase.VOTE
            gs._step = 0
        else:
            gs._phase = Phase.DEBATE

        return {"game": gs}

    def vote_node(self, state: GraphState, config: RunnableConfig) -> GraphState:
        """All alive players cast a vote simultaneously to exile someone."""
        gs = state["game"]
        player_objects: dict[str, BasePlayer] = config.get("configurable", {}).get("player_objects", {})

        round_log = gs._game_summary_log.setdefault(gs._round_num, {})
        debate_log: list[dict] = round_log.setdefault("day_debate", [])
        votes: list[dict] = []

        # 1. Run voting in parallel for all alive players
        with ThreadPoolExecutor(max_workers=len(gs._alive_players)) as executor:
            futures = {
                name: executor.submit(player_objects[name].vote, gs._alive_players, debate_log, gs._round_num)
                for name in gs._alive_players
            }

            for name, future in futures.items():
                target, raw_output = future.result()

                # Extract the public reasoning
                reasoning = raw_output.get("reasoning", "No public reason provided.")

                # Create a dictionary for THIS specific vote, then append it to the list
                vote_data = {
                    "voter": name,
                    "vote_for": target,
                    "reasoning": f"{name} voted to exile {target}. Reason: {reasoning}",
                }
                votes.append(vote_data)

        # Safely initialize the round log dictionary before adding votes
        round_log = gs._game_summary_log.setdefault(gs._round_num, {})
        round_log["votes"] = votes

        # 2. Tally the votes
        vote_counts = defaultdict(int)
        for vote in votes:
            target = vote["vote_for"]
            # Defensive check: ensure target isn't None, empty, or the literal string "None"
            if target and (str(target).lower() not in ["none", ""]):
                vote_counts[target] += 1

        # 3. Determine the outcome
        threshold = math.ceil(len(gs._alive_players) / 2.0)

        if vote_counts:
            max_votes = max(vote_counts.values())
            # Find everyone who tied for the highest number of votes
            tied_players = [p for p, c in vote_counts.items() if c == max_votes]

            if max_votes >= threshold:
                exiled_player = random.choice(tied_players)
            else:
                exiled_player = None
        else:
            exiled_player = None
            max_votes = 0
            tied_players = []

        # 4. Print the dramatic results to your terminal
        tqdm.tqdm.write("\n=== VOTING RESULTS ===")
        for vote in votes:
            tqdm.tqdm.write(f"• {vote['reasoning']}")

        if exiled_player:
            announcement = f"=> {exiled_player} received {max_votes} votes. The player will be exiled!"
        else:
            if max_votes > 0:
                announcement = f"=> The highest vote count was {max_votes} for {' and '.join(tied_players)} (Require at least {threshold} vote{'s' if threshold > 1 else ''}). Not enough consensus. No one is exiled."
            else:
                announcement = "=> No valid votes were cast. No one is exiled."

        tqdm.tqdm.write(announcement)

        # Broadcast the announcement to ALL currently alive players including one who is going to be exiled
        for name in gs._alive_players:
            if name in player_objects:
                player_objects[name].receive_announcement(gs._round_num, "Day", announcement)

        # 5. Mutate State manually
        gs._exiled = exiled_player
        gs._step = 0

        # Advance phase
        gs._phase = Phase.EXILE

        return {"game": gs}

    def exile_node(self, state: GraphState, config: RunnableConfig) -> GraphState:
        """Apply the vote outcome, broadcast announcement, and flush notes to disk."""
        gs = state["game"]
        player_objects: dict[str, BasePlayer] = config.get("configurable", {}).get("player_objects", {})

        if gs._exiled:
            # Safely remove the player from the alive list
            gs._alive_players = [p for p in gs._alive_players if p != gs._exiled]

        # Log the exiled player
        gs._game_summary_log[gs._round_num]["day_exiled"] = gs._exiled

        # Advance State
        gs._phase = Phase.CHECK_WINNER_DAY

        return {"game": gs}

    def check_winner_day_node(self, state: GraphState, config: RunnableConfig) -> GraphState:
        gs = state["game"]
        player_objects: dict[str, BasePlayer] = config.get("configurable", {}).get("player_objects", {})

        winner = self._compute_current_winner(gs)
        if not winner:
            gs._phase = Phase.ELIMINATE
            gs._round_num += 1  # Advance to the next round (Night phase)
            game_status = "After voting, the game continues."
        else:
            gs._phase = Phase.END
            game_status = "After voting, the game ends."

        analyzers = gs._alive_players.copy()
        if gs._exiled and gs._exiled not in analyzers:
            analyzers.append(gs._exiled)

        current_vote_log: list = []
        votes_data = gs._game_summary_log.get(gs._round_num, {}).get("votes", [])
        for vote in votes_data:
            current_vote_log.append(vote["reasoning"])

        with ThreadPoolExecutor(max_workers=len(analyzers)) as executor:
            threads = [
                executor.submit(
                    player_objects[name].update_suspicion_from_vote,
                    current_vote_log,
                    gs._exiled,
                    game_status,
                )
                for name in analyzers
                if name in player_objects
            ]

            # Pause the game right here until every single thread finishes its work!
            for thread in threads:
                thread.result()  # We don't save the result, we just wait for it to finish.

        gs._step = 0

        return {"game": gs}

    def end_node(self, state: GraphState, config: RunnableConfig) -> GraphState:
        gs = state["game"]
        player_objects: dict[str, BasePlayer] = config.get("configurable", {}).get("player_objects", {})
        coach_object: Coach | None = config.get("configurable", {}).get("coach", None)

        scenario = config.get("configurable", {}).get("scenario", "baseline")
        scenario_config = SCENARIO_CONFIG[scenario]

        tqdm.tqdm.write(f"\n=== GAME OVER ===\nWinner: {gs._winner}")

        # ==========================================
        # FORMATTING LOGS TO MARKDOWN
        # ==========================================

        # --- A. Format Game Summary ---

        # Add roles to game summary
        role_log = gs._game_summary_log.setdefault("roles", {})
        for name, role in gs._roles.items():
            role_log[name] = role.value

        game_dir = (
            Path(__file__).parent
            / "game_logs"
            / scenario
            / f"game_{config.get('configurable', {}).get('game_id', 'unknown')}"
        ).resolve()

        json_string = json.dumps(gs._game_summary_log, indent=4)
        write_to_file(
            game_dir / GAME_SUMMARY_FILENAME.replace(".md", ".json"), json_string
        )  # Store game_summary as json

        summary_lines = []
        winner = gs._game_summary_log.get("winner", "Unknown")
        summary_lines.append(f"# Game Summary\n**Winner:** {winner}\n")

        # Log roles
        summary_lines.append("# Roles Assigned")
        for name, role in gs._roles.items():
            summary_lines.append(f"**{name}:** {role.value}")
        summary_lines.append("\n")

        for round_num, data in gs._game_summary_log.items():
            if str(round_num) == "winner":
                continue

            summary_lines.append(f"## Round {round_num}")

            # Night Actions
            night_actions = data.get("night_actions", [])
            summary_lines.append("**Night Actions:**")
            for action in night_actions:
                summary_lines.append(f"• {action}")

            # Night Outcomes
            night_kills = data.get("night_eliminated", [])
            summary_lines.append(f"**Night Eliminations:** {', '.join(night_kills) if night_kills else 'None'}")

            # Day Debate
            summary_lines.append("\n### Day Debate")
            for debate_entry in data.get("day_debate", []):
                speaker = debate_entry.get("speaker", "Unknown")
                statement = debate_entry.get("statement", "")
                summary_lines.append(f"**{speaker}:** {statement}")

            # Votes
            summary_lines.append("\n### Votes")
            for vote in data.get("votes", []):
                voter = vote.get("voter", "Unknown")
                vote_for = vote.get("vote_for", "None")
                reason = vote.get("reasoning", "No reason provided.")
                summary_lines.append(f"**{voter}** voted for **{vote_for}**\n> {reason}")

            # Day Exiled
            summary_lines.append(f"\n**Day Exiled:** {data.get('day_exiled', 'None')}")
            summary_lines.append("---\n")

        formatted_summary_md = "\n".join(summary_lines)

        # 2. Generate coach's feedback and update coach strategy
        if scenario_config.get("coaching"):
            if coach_object:
                tqdm.tqdm.write("=> Coach is reviewing the game and writing feedback...")
                # Pass the beautifully formatted markdown string to the coach!
                coach_object.run(formatted_summary_md)
            else:
                tqdm.tqdm.write("=> WARNING: Coaching is enabled, but no coach object was provided in config.")

        # 3. Update and write out strategy for ALL players
        tqdm.tqdm.write("=> Players are analyzing their performance and updating strategies...")
        with ThreadPoolExecutor(max_workers=len(player_objects)) as executor:
            threads = [
                executor.submit(player.update_strategy, formatted_summary_md) for player in player_objects.values()
            ]
            for thread in threads:
                thread.result()

        # 4. Write per-game log files to disk
        player_night_action_log_dir = (game_dir / "player_night_action_log").resolve()
        player_note_dir = (game_dir / "player_note").resolve()

        # Ensure the directories exist
        game_dir.mkdir(parents=True, exist_ok=True)
        player_night_action_log_dir.mkdir(parents=True, exist_ok=True)
        player_note_dir.mkdir(parents=True, exist_ok=True)

        # Write Game Summary
        write_to_file(game_dir / GAME_SUMMARY_FILENAME, formatted_summary_md)

        # --- B. Format Secret Action Logs ---

        # Wolf Debate Log
        wolf_lines = ["# Wolf Debate and Eliminations"]
        for entry in gs._wolf_debate_log:
            round_number, speaker_name, statement, target, analysis = entry
            wolf_lines.append(f"### Round {round_number}")
            wolf_lines.append(f"**Speaker:** {speaker_name}")
            wolf_lines.append(f"**Statement:** {statement}")
            wolf_lines.append(f"**Target Proposal:** {target}")
            wolf_lines.append(f"**Private Analysis:** {analysis}")
            wolf_lines.append("---")
        write_to_file(
            player_night_action_log_dir / WOLF_DEBATE_LOG_FILENAME,
            "\n".join(wolf_lines) if len(wolf_lines) > 1 else "No wolf debate recorded.",
        )

        # Wolf Target Log
        wolf_target_lines = ["# Wolf Targets"]
        for entry in gs._wolf_target_log:
            round_number, target = entry
            wolf_target_lines.append(f"**Round {round_number}:** Target **{target}**\n---")
        write_to_file(
            player_night_action_log_dir / WOLF_TARGET_LOG_FILENAME,
            "\n".join(wolf_target_lines) if len(wolf_target_lines) > 1 else "No wolf target log.",
        )

        # Guard Log
        guard_lines = ["# Guard Actions"]
        for entry in gs._guard_log:
            round_number, target, analysis = entry
            guard_lines.append(f"**Round {round_number}:** Protected **{target}**\n> Analysis: {analysis}\n---")
        write_to_file(
            player_night_action_log_dir / GUARD_LOG_FILENAME,
            "\n".join(guard_lines) if len(guard_lines) > 1 else "No guard actions.",
        )

        # Seer Log
        seer_lines = ["# Seer Actions"]
        for entry in gs._seer_log:
            round_number, target, is_wolf, analysis = entry
            seer_lines.append(
                f"**Round {round_number}:** Investigated **{target}** (Result: {'WOLF' if is_wolf else 'Villager'})\n> Analysis: {analysis}\n---"
            )
        write_to_file(
            player_night_action_log_dir / SEER_LOG_FILENAME,
            "\n".join(seer_lines) if len(seer_lines) > 1 else "No seer actions.",
        )

        # Witch Log
        witch_lines = ["# Witch Actions"]
        for entry in gs._witch_log:
            round_number, saved, poisoned, analysis = entry
            witch_lines.append(
                f"**Round {round_number}:** Saved **{saved}** | Poisoned **{poisoned}**\n> Analysis: {analysis}\n---"
            )
        write_to_file(
            player_night_action_log_dir / WITCH_LOG_FILENAME,
            "\n".join(witch_lines) if len(witch_lines) > 1 else "No witch actions.",
        )

        # --- C. Format Player Notes ---

        for name, player in player_objects.items():
            role_name = player._role.value
            write_to_file(player_note_dir / PLAYER_NOTE_FILENAME.format(name=name, role=role_name), str(player._note))

        tqdm.tqdm.write("=> Game successfully wrapped up. Logs saved to disk. Ready for the next round!")

        # --- D. Format Suspicion Log ---
        suspicion_lines = ["# Game Suspicion Log\n"]

        # Sort the rounds to guarantee chronological order
        for round_num in sorted(gs._suspicion_log.keys()):
            round_data = gs._suspicion_log[round_num]
            suspicion_lines.append(f"## Round {round_num}")

            for player_name, suspicions in round_data.items():
                # If you want to include their role, you can pull it from gs._roles if available:
                role_str = gs._roles[player_name].value if player_name in gs._roles else "Unknown"
                suspicion_lines.append(f"### {player_name} ({role_str})'s Suspicions")

                if not suspicions:
                    suspicion_lines.append("No suspicions recorded.\n")
                    continue

                for target_name, data in suspicions.items():
                    score = data.get("score", "N/A")
                    reason = data.get("reason", "No reason provided.")
                    suspicion_lines.append(f"- **{target_name}** (Score: {score}): {reason}")

                suspicion_lines.append("")  # Adds a blank line for readability

            suspicion_lines.append("---\n")  # Visual divider between rounds

        # Save to disk
        write_to_file(game_dir / SUSPICION_LOG_FILENAME, "\n".join(suspicion_lines))
        json_string = json.dumps(gs._suspicion_log, indent=4)
        write_to_file(game_dir / SUSPICION_LOG_FILENAME.replace(".md", ".json"), json_string)

        return {"game": gs}

    def build_graph(self):
        graph = StateGraph(GraphState)

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
        graph.add_node("end", self.end_node)

        graph.set_entry_point("eliminate")

        graph.add_conditional_edges("eliminate", lambda s: s["game"]._phase.value)
        graph.add_conditional_edges("protect", lambda s: s["game"]._phase.value)
        graph.add_conditional_edges("unmask", lambda s: s["game"]._phase.value)
        graph.add_conditional_edges("save_or_poison", lambda s: s["game"]._phase.value)
        graph.add_conditional_edges("resolve_night", lambda s: s["game"]._phase.value)
        graph.add_conditional_edges("check_winner_night", lambda s: s["game"]._phase.value)
        graph.add_conditional_edges("debate", lambda s: s["game"]._phase.value)
        graph.add_conditional_edges("vote", lambda s: s["game"]._phase.value)
        graph.add_conditional_edges("exile", lambda s: s["game"]._phase.value)
        graph.add_conditional_edges("check_winner_day", lambda s: s["game"]._phase.value)
        graph.add_edge("end", END)

        return graph.compile()


class GraphState(TypedDict):
    game: GameState
