"""
run.py - Werewolf experiment runner

Usage
-----
Start fresh (overwrite everything from game_001):
    python run.py --scenario baseline --mode override

Resume from where you left off:
    python run.py --scenario baseline --mode append

Override number of games:
    python run.py --scenario baseline --mode override --games 10

File layout produced
--------------------
game_logs/
  {scenario}/
    results_summary.csv
    game_001/
      game_summary.md
      player_strategies/   snapshot of strategies at game start
      ...

strategies/
  {scenario}/
    player_1_strategy.txt   slot 0 (villager side)
    player_2_strategy.txt   slot 1 (villager side)
    ...
    player_6_strategy.txt   slot 5 (wolf side)
    player_7_strategy.txt   slot 6 (wolf side)
    coach_strategy.txt

Design note — why player_id, not name
--------------------------------------
Player names are re-randomised every game, so keying strategies by name would produce a fresh strategy file for every new name.  Instead each physical "seat"
at the table is assigned a stable player_id (1 … 7). The villager seats (1-5) rotate roles in round-robin order; the wolf seats (6-7) are always wolves.
Strategy files are keyed by player_id so learning accumulates across games regardless of the cosmetic name.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, cast
import argparse
import shutil
from pathlib import Path
import random

from game import GameState
from config import SCENARIO_CONFIG
from players.guard import Guard
from players.seer import Seer
from players.witch import Witch
from players.villager import Villager
from players.wolf import Wolf
from players.coach import Coach
from players.base_player import Role, WOLF_SIDE
from utils import get_llm
from dotenv import load_dotenv
import pandas as pd
from constants import RESULTS_SUMMARY_FILENAME

load_dotenv()

if TYPE_CHECKING:
    from players.base_player import BasePlayer


# ============================================================
# Name pool & role configuration
# ============================================================

NAMES = [
    "Liam", "Emma", "Noah", "Olivia", "Mason", "Sophia", "Elijah", "Ava",
    "Lucas", "Mia", "Oliver", "Isabella", "Ethan", "Charlotte", "Aiden",
    "Amelia", "Jacob", "Harper", "Logan", "Evelyn", "James", "Abigail",
    "Benjamin", "Emily", "Carter", "Elizabeth", "Alexander", "Sofia",
    "Daniel", "Avery", "Michael", "Ella", "Henry", "Madison", "Jackson",
    "Scarlett", "Sebastian", "Victoria", "Matthew", "Aria", "Samuel",
    "Grace", "David", "Chloe", "Joseph", "Camila", "John", "Penelope",
    "Owen", "Riley", "Wyatt", "Layla", "Jack", "Lillian", "Luke", "Nora",
    "Jayden", "Zoe", "Dylan", "Hannah", "Grayson", "Nevaeh", "Levi",
    "Lily", "Isaac", "Violet", "Gabriel", "Stella", "Julian", "Aurora",
    "Mateo", "Savannah", "Anthony", "Claire", "Lincoln", "Skylar",
    "Joshua", "Mila", "Christopher", "Ellie", "Andrew", "Addison",
    "Theodore", "Eleanor", "Caleb", "Quinn", "Ryan", "Caroline", "Asher",
    "Brooklyn", "Nathan", "Samantha", "Thomas", "Maya", "Leo", "Naomi",
    "Isaiah", "Audrey", "Charles", "Lucy",
]

# Stable slot counts — never change these without also clearing strategy files.
N_VILLAGER_SLOTS = 5
N_WOLF_SLOTS     = 2
N_SLOTS          = N_VILLAGER_SLOTS + N_WOLF_SLOTS   # 7 total

# Villager role pool — the round-robin rotates this list one position each game.
# Length must equal N_VILLAGER_SLOTS.
VILLAGER_ROLE_POOL: list[Role] = (
    [Role.SEER]     * 1
    + [Role.GUARD]  * 1
    + [Role.WITCH]  * 1
    + [Role.VILLAGER] * 2
)
assert len(VILLAGER_ROLE_POOL) == N_VILLAGER_SLOTS

ROLE_TO_CLASS: dict[Role, type] = {
    Role.VILLAGER: Villager,
    Role.WEREWOLF: Wolf,
    Role.SEER:     Seer,
    Role.GUARD:    Guard,
    Role.WITCH:    Witch,
}

MAX_DEBATE_TURNS = 6


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Werewolf game experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-s", "--scenario", type=str, default="baseline",
                        choices=list(SCENARIO_CONFIG.keys()),
                        help="Experiment scenario to run")
    parser.add_argument("-g", "--games",   type=int, default=100,
                        help="Total number of games to run")
    parser.add_argument("-m", "--mode",    type=str, default="append",
                        choices=["override", "append"],
                        help=("override: wipe existing results and start from game_001. "
                              "append: detect the last completed game and continue."))
    return parser.parse_args()


# ============================================================
# Append-mode: detect the last completed game
# ============================================================

def _last_completed_game(scenario: str) -> int:
    """Return the highest game number already completed for this scenario.

    A folder is considered complete when it contains game_summary.md.
    Returns 0 if none are found.
    """
    log_root = (Path(__file__).parent / "game_logs" / scenario).resolve()
    if not log_root.exists():
        return 0

    completed = []
    for folder in log_root.iterdir():
        if folder.is_dir() and folder.name.startswith("game_"):
            if (folder / "game_summary.md").exists():
                try:
                    completed.append(int(folder.name.split("_")[1]))
                except (IndexError, ValueError):
                    pass

    return max(completed, default=0)


# ============================================================
# Role assignment — random names + round-robin roles
# ============================================================

def assign_game() -> tuple[list[str], dict[str, Role], dict[str, int]]:
    """Randomly assign names to the N_SLOTS seats and rotate villager roles.

    Round-robin: VILLAGER_ROLE_POOL is advanced by one position each call so that each villager seat experiences every role over N_VILLAGER_SLOTS games.
    Wolf seats are always Role.WEREWOLF.

    Returns
    -------
    players : list[str]
        All N_SLOTS player names in order [villager names..., wolf names...].
    roles : dict[str, Role]
        {name: role} for all players this game.
    player_ids : dict[str, int]
        {name: 1...7} — stable slot identifier used for strategy files.
        1 … 5 are villager slots; 6 … 7 are wolf slots.
    """
    global VILLAGER_ROLE_POOL

    # Rotate roles before sampling names so game 1 uses the original order
    VILLAGER_ROLE_POOL = VILLAGER_ROLE_POOL[1:] + VILLAGER_ROLE_POOL[:1]

    # Draw N_SLOTS unique names without replacement
    villager_names: list[str] = random.sample(NAMES, N_VILLAGER_SLOTS)
    remaining_names = [n for n in NAMES if n not in villager_names]
    wolf_names: list[str] = random.sample(remaining_names, N_WOLF_SLOTS)

    players = villager_names + wolf_names

    # Assign roles — villager seats get the rotated pool; wolf seats get WEREWOLF
    roles: dict[str, Role] = {}
    for name, role in zip(villager_names, VILLAGER_ROLE_POOL):
        roles[name] = role
    for name in wolf_names:
        roles[name] = Role.WEREWOLF

    # Stable slot IDs — pure integers starting from 1
    player_ids: dict[str, int] = {
        name: idx + 1
        for idx, name in enumerate(players)
    }

    # Randomly shuffle the players (to avoid if there is any case the LLM remembers the order of the roles)
    random.shuffle(players)

    return players, roles, player_ids


# ============================================================
# Player construction
# ============================================================

def build_player_objects(
    players:    list[str],
    roles:      dict[str, Role],
    player_ids: dict[str, int],
    villager_llm,
    wolf_llm,
    game_id:  str,
    scenario: str,
) -> dict[str, "BasePlayer"]:
    """Instantiate one player object per seat.

    Villager-side seats get the small model; wolf seats get the large model.
    Objects are created fresh every game (name and role may have changed).
    Strategy files are loaded at __init__ via player_id, so learning persists.
    """
    player_objects: dict[str, "BasePlayer"] = {}
    
    for name in players:
        role = roles[name]
        player_id = player_ids[name]
        model = wolf_llm if role in WOLF_SIDE else villager_llm
        cls = ROLE_TO_CLASS[role]
        player_objects[name] = cls(
            player_id=player_id,
            name=name,
            role=role,
            model=model,
            game_id=game_id,
            scenario=scenario,
        )

    # Seed suspicion scores
    actual_wolves = [p for p, r in roles.items() if r in WOLF_SIDE]
    for name, player_obj in player_objects.items():
        other_players = [p for p in players if p != name]
        if player_obj._role in WOLF_SIDE:
            player_obj.init_suspicions(other_players, wolf_teammates=actual_wolves)
        else:
            player_obj.init_suspicions(other_players)

    return player_objects


# ============================================================
# Single game
# ============================================================

def run_game(
    players:    list[str],
    roles:      dict[str, Role],
    player_ids: dict[str, int],
    villager_llm,
    wolf_llm,
    game_id:  str,
    scenario: str,
) -> GameState:
    """Build player objects, run one game, return the final GameState.

    Raises immediately on any exception — caller decides how to handle it.
    """
    player_objects = build_player_objects(players, roles, player_ids, villager_llm, wolf_llm, game_id, scenario)
    coach = Coach(model=wolf_llm, game_id=game_id, scenario=scenario)

    # Derive role-specific lists from the per-game roles dict, not a global
    seer       = next((p for p in players if roles[p] == Role.SEER),  None)
    guard      = next((p for p in players if roles[p] == Role.GUARD), None)
    witch      = next((p for p in players if roles[p] == Role.WITCH), None)
    villagers  = [p for p in players if roles[p] == Role.VILLAGER]
    werewolves = [p for p in players if roles[p] == Role.WEREWOLF]

    initial_state = GameState(
        round_num=1,
        players=players,
        alive_players=players.copy(),
        villagers=villagers,
        werewolves=werewolves,
        seer=seer,
        guard=guard,
        witch=witch,
        roles=roles,
    )

    runnable = initial_state.build_graph()
    final_state = runnable.invoke(
        {"game": initial_state},
        config={
            "recursion_limit": 1000,
            "configurable": {
                "player_objects":   player_objects,
                "MAX_DEBATE_TURNS": MAX_DEBATE_TURNS,
                "scenario":         scenario,
                "game_id":          game_id,
                "coach":            coach,
            },
        },
    )
    return cast(GameState, final_state["game"])


# ============================================================
# Multi-game experiment loop
# ============================================================

def run(scenario: str = "baseline", num_games: int = 100, mode: str = "append") -> None:
    """Run num_games games back-to-back under the given scenario.

    mode="override"  Start from game_001, overwriting any existing results.
    mode="append"    Detect the last completed game and continue from there.

    Any exception inside a game stops the program immediately.
    """
    if mode == "append":
        last_done  = _last_completed_game(scenario)
        start_from = last_done + 1
        if last_done > 0:
            print(f"\nAppend mode: resuming from game {start_from} "
                  f"({last_done} games already completed).")
    elif mode == "override":
        start_from = 1
        print("\nOverride mode: starting fresh from game_001.")
    else:
        raise ValueError("Mode can only be 'append' or 'override'.")

    VILLAGER_MODEL = SCENARIO_CONFIG[scenario]["villager_model"]
    WOLF_MODEL     = SCENARIO_CONFIG[scenario]["wolf_model"]

    print(f"\n{'=' * 62}")
    print(f"  EXPERIMENT : {scenario}")
    print(f"  Games      : {start_from:03d} → {start_from + num_games - 1:03d}  ({num_games} to run)")
    print(f"  Villager model : {VILLAGER_MODEL}")
    print(f"  Wolf model     : {WOLF_MODEL}")
    print(f"{'=' * 62}\n")

    villager_llm = get_llm(VILLAGER_MODEL)
    wolf_llm     = get_llm(WOLF_MODEL)

    out_path = (Path(__file__).parent / "game_logs" / scenario / RESULTS_SUMMARY_FILENAME).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for i in range(start_from, start_from + num_games):
        game_id = f"{i:03d}"

        # ── Snapshot current strategies before this game modifies them ──
        strategies_src = (Path(__file__).parent / "strategies" / scenario).resolve()
        player_strategies_dst = (
            Path(__file__).parent / "game_logs" / scenario / f"game_{game_id}" / "player_strategies"
        ).resolve()
        player_strategies_dst.mkdir(parents=True, exist_ok=True)
        if strategies_src.exists():
            for f in strategies_src.iterdir():
                if f.is_file():
                    shutil.copy2(f, player_strategies_dst / f.name)

        # ── Assign names and roles for this game ────────────────────────
        players, roles, player_ids = assign_game()

        role_summary = ", ".join(f"{n}={r.value}" for n, r in roles.items())
        id_summary   = ", ".join(f"{n}={pid}" for n, pid in player_ids.items())
        print(f"\n--- Game {i:3d} / {start_from + num_games - 1}  [{game_id}] ---")
        print(f"    Roles : {role_summary}")

        # No try/except — any error stops the program immediately
        final_state = run_game(
            players, roles, player_ids,
            villager_llm, wolf_llm,
            game_id, scenario,
        )

        winner       = final_state._winner
        total_rounds = final_state._round_num
        assert winner is not None, "Game ended without a winner — check end_node."

        print(f"    Winner: {winner}  (rounds: {total_rounds})")

        # ── Append one row to the CSV results file ──────────────────────
        result = {
            "game_id":      game_id,
            "winner":       winner,
            "rounds":       total_rounds,
            "roles":        role_summary,
            "player_ids":   id_summary,
        }
        pd.DataFrame([result]).to_csv(
            out_path,
            mode="a",
            index=False,
            header=not out_path.is_file(),
        )
        print(f"    Logged → {out_path}")


# ============================================================
# Entry point
# ============================================================

import numpy as np
players = tied_players = []
N_REPLICATES = 10

def game_rng(replicate, game_id, master=42):
    ss = np.random.SeedSequence([master, replicate, game_id])
    return np.random.default_rng(ss)

for replicate in range(1, N_REPLICATES + 1):   # e.g. 1..10
    for game_id in range(1, 101):
        rng = game_rng(replicate, game_id)   # fresh, deterministic per game
        seating = rng.permutation(players)
        tie_loser = rng.choice(tied_players)
        ...

if __name__ == "__main__":
    args = parse_args()
    print(args)
    run(scenario=args.scenario, num_games=args.games, mode=args.mode)