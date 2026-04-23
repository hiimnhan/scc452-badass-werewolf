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
    
/opt/miniconda3/envs/scc452-badass-werewolf/bin/python run.py --scenario coach_and_self_analyze --mode append --games 3

File layout produced
--------------------
game_logs/
  {scenario}/
    results_summary.txt
    game_001/
      game_summary.txt        all public announcements in order
      debate_log.txt          every statement from every day debate
      vote_log.txt            every vote cast across all rounds
      roles_this_game.txt     who played which role
      {name}_{role}_note.txt  each player's compiled note (game summary + suspicions)
      coach_feedback.txt      coach's post-game feedback (if coaching enabled)

strategies/
  {scenario}/
    {name}_strategy.txt  each player's accumulated strategy
    coach_strategy.txt          coach's accumulated coaching strategy
"""

from __future__ import annotations
from typing import TYPE_CHECKING, cast
import argparse
import shutil
from pathlib import Path
import random

from game import GameState
from config import SCENARIO_CONFIG, VILLAGER_MODEL, WOLF_MODEL
from players.guard import Guard
from players.seer import Seer
from players.witch import Witch
from players.villager import Villager
from players.wolf import Wolf
from players.coach import Coach
from players.base_player import Role, WOLF_SIDE
from utils import get_llm, write_to_file
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()  # Load environment variables from .env

if TYPE_CHECKING:
    from players.base_player import BasePlayer

# ============================================================
# Player roster & role pool
# ============================================================

VILLAGER_PLAYERS: list[str] = ["Nhan", "Cong", "Nam", "Jin", "Hoang"]
WOLF_PLAYERS: list[str] = ["Riley", "Meg"]
PLAYERS: list[str] = VILLAGER_PLAYERS + WOLF_PLAYERS

# Villager role pool — reshuffled randomly before every game.
# Must equal len(VILLAGER_PLAYERS).
VILLAGER_ROLE_POOL: list[Role] = [Role.SEER] * 1 + [Role.GUARD] * 1 + [Role.WITCH] * 1 + [Role.VILLAGER] * 2
# random.shuffle(VILLAGER_ROLE_POOL)

assert len(VILLAGER_ROLE_POOL) == len(VILLAGER_PLAYERS), (
    f"VILLAGER_ROLE_POOL has {len(VILLAGER_ROLE_POOL)} entries but VILLAGER_PLAYERS has {len(VILLAGER_PLAYERS)}."
)

ROLE_TO_CLASS: dict[Role, type] = {
    Role.VILLAGER: Villager,
    Role.WEREWOLF: Wolf,
    Role.SEER: Seer,
    Role.GUARD: Guard,
    Role.WITCH: Witch,
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
    parser.add_argument(
        "-s",
        "--scenario",
        type=str,
        default="baseline",
        choices=list(SCENARIO_CONFIG.keys()),
        help="Experiment scenario to run",
    )
    parser.add_argument(
        "-g",
        "--games",
        type=int,
        default=100,
        help="Total number of games to run",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="append",
        choices=["override", "append"],
        help=(
            "override: wipe existing results and start from game_001. "
            "append:   detect the last completed game and continue from there."
        ),
    )
    return parser.parse_args()


# ============================================================
# Append-mode: detect the last completed game number
# ============================================================


def _last_completed_game(scenario: str) -> int:
    """Return the highest game number already completed for this scenario.

    A game folder is considered complete when it contains game_summary.txt.
    Returns 0 if no completed games are found (i.e. start from game_001).
    """
    log_root = (Path(__file__).parent / "game_logs" / scenario).resolve()
    if not log_root.exists():
        return 0

    completed = []
    for folder in log_root.iterdir():
        if not folder.is_dir():
            continue
        # Folder names: game_001, game_002, ...
        if folder.name.startswith("game_") and (folder / "game_summary.md").exists():
            try:
                completed.append(int(folder.name.split("_")[1]))
            except (IndexError, ValueError):
                pass

    return max(completed, default=0)


# ============================================================
# Role assignment
# ============================================================


def assign_roles_round_robin() -> dict[str, Role]:
    """Rotate villager roles by one position each game (round-robin).
    Wolves are always wolves — only villager-side roles rotate.

    Example over 5 games (5 villager players, 3 roles):
      Game 1: Alice=Seer,  Bob=Guard, Selena=Witch, Raj=Villager, Frank=Villager
      Game 2: Alice=Guard, Bob=Witch, Selena=Villager, Raj=Villager, Frank=Seer
      Game 3: Alice=Witch, Bob=Villager, Selena=Villager, Raj=Seer, Frank=Guard
      ... and so on
    """
    global VILLAGER_ROLE_POOL
    random.shuffle(VILLAGER_ROLE_POOL)
    VILLAGER_ROLE_POOL = VILLAGER_ROLE_POOL[1:] + VILLAGER_ROLE_POOL[:1]
    return dict(zip(PLAYERS, VILLAGER_ROLE_POOL + ([Role.WEREWOLF] * len(WOLF_PLAYERS))))


# ============================================================
# Player construction
# ============================================================


def build_player_objects(
    roles: dict[str, Role],
    villager_llm,
    wolf_llm,
    game_id: str,
    scenario: str,
) -> dict[str, BasePlayer]:
    """Instantiate one player object per player for the given role assignment.

    Villager-side players get the small model; wolves get the large model.
    Objects are created fresh every game because the role (and class) can change.
    Strategies persist on disk and are loaded automatically at __init__.
    """
    player_objects: dict[str, BasePlayer] = {}
    for name in PLAYERS:
        role = roles[name]
        model = wolf_llm if role in WOLF_SIDE else villager_llm
        cls = ROLE_TO_CLASS[role]
        player_objects[name] = cls(
            name=name,
            role=role,
            model=model,
            game_id=game_id,
            scenario=scenario,
        )

    # Find out who the actual wolves are in THIS specific game
    actual_wolves = [p for p, r in roles.items() if r in WOLF_SIDE]

    for name, player_obj in player_objects.items():
        other_players = [p for p in PLAYERS if p != name]

        if player_obj._role in WOLF_SIDE:
            # Wolves get the dynamic list of their teammates
            player_obj.init_suspicions(other_players, wolf_teammates=actual_wolves)
        else:
            # Villagers get no one
            player_obj.init_suspicions(other_players)

    return player_objects


# ============================================================
# Single game
# ============================================================


def run_game(
    villager_llm,
    wolf_llm,
    roles: dict[str, Role],
    game_id: str,
    scenario: str,
) -> GameState:
    """Build player objects, run one game, return the final GameState.

    Raises immediately on any exception — caller decides how to handle it.
    """
    player_objects = build_player_objects(roles, villager_llm, wolf_llm, game_id, scenario)
    # Coach uses villager_llm — it coaches the villager side, not the wolves
    coach = Coach(model=wolf_llm, game_id=game_id, scenario=scenario)

    seer = next((p for p in PLAYERS if roles[p] == Role.SEER), None)
    guard = next((p for p in PLAYERS if roles[p] == Role.GUARD), None)
    witch = next((p for p in PLAYERS if roles[p] == Role.WITCH), None)
    villagers = [p for p in PLAYERS if roles[p] == Role.VILLAGER]
    werewolves = [p for p in PLAYERS if roles[p] == Role.WEREWOLF]

    initial_state = GameState(
        round_num=1,
        players=PLAYERS,
        alive_players=PLAYERS.copy(),
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
                "player_objects": player_objects,
                "MAX_DEBATE_TURNS": MAX_DEBATE_TURNS,
                "scenario": scenario,
                "game_id": game_id,
                "coach": coach,
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
                     If all games are already done, prints a message and exits.

    Any exception inside a game stops the program immediately.
    """
    # ── Determine starting game number ────────────────────────────────
    if mode == "append":
        last_done = _last_completed_game(scenario)
        start_from = last_done + 1
        # if start_from > num_games:
        #     print(
        #         f"\nNothing to do: {last_done} games already completed for scenario '{scenario}' (target: {num_games})."
        #     )
        #     return
        if last_done > 0:
            print(f"\nAppend mode: resuming from game {start_from} ({last_done} games already completed).")

    elif mode == "override":
        start_from = 1

        print("\nOverride mode: starting fresh from game_001.")
    else:
        raise ValueError("Mode can only be either append or override.")

    games_to_run = num_games

    print(f"\n{'=' * 62}")
    print(f"  EXPERIMENT : {scenario}")
    print(f"  Games      : {start_from:03d} → {start_from + num_games - 1:03d}  ({games_to_run} to run)")
    print(f"  Villager model : {VILLAGER_MODEL}")
    print(f"  Wolf model     : {WOLF_MODEL}")
    print(f"{'=' * 62}\n")

    # Build LLM instances once — reused across all games
    villager_llm = get_llm(VILLAGER_MODEL)
    wolf_llm = get_llm(WOLF_MODEL)

    results: list[dict] = []

    for i in range(start_from, start_from + num_games):
        game_id = f"{i:03d}"
        strategies_src = (Path(__file__).parent / "strategies" / scenario).resolve()
        player_strategies_dst = (
            Path(__file__).parent / "game_logs" / scenario / f"game_{game_id}" / "player_strategies"
        ).resolve()
        player_strategies_dst.mkdir(parents=True, exist_ok=True)
        if strategies_src.exists():
            for f in strategies_src.iterdir():
                if f.is_file():
                    shutil.copy2(f, player_strategies_dst / f.name)

        roles = assign_roles_round_robin()
        role_summary = ", ".join(f"{n}={r.value}" for n, r in roles.items())
        print(f"\n--- Game {i:3d} / {start_from + num_games - 1}  [{game_id}] ---")
        print(f"    Roles: {role_summary}")

        # No try/except — any error stops the program immediately
        final_state = run_game(villager_llm, wolf_llm, roles, game_id, scenario)
        winner = final_state._winner
        total_rounds = final_state._round_num
        assert winner is not None

        # results.append({"game_id": game_id, "winner": winner, "roles": roles})
        print(f"    Winner: {winner}")
        result = {"game_id": game_id, "winner": winner, "rounds": total_rounds, "roles": role_summary}
        out_path = (Path(__file__).parent / "game_logs" / scenario / "results_summary.csv").resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        new_data_df = pd.DataFrame([result])
        file_exists = out_path.is_file()

        new_data_df.to_csv(
            out_path,
            mode="a",
            index=False,
            header=not file_exists,  # Only write header if it's a new file
        )

        print(f"\n  Results saved to: {out_path}")

    # ── Final summary ──────────────────────────────────────────────────
    # print(f"\n{'=' * 62}")
    # print(f"  RESULTS — {scenario}  (games {start_from:03d}-{start_from + num_games - 1:03d})")
    # print(f"{'=' * 62}")

    # villager_wins = sum(1 for r in results if r["winner"] == "Villagers")
    # wolf_wins = sum(1 for r in results if r["winner"] == "Werewolves")

    # for r in results:
    #     print(f"  {r['game_id']}: {r['winner']}")

    # print(f"\n  Villagers  : {villager_wins:3d} wins  ({villager_wins / games_to_run * 100:.1f}%)")
    # print(f"  Werewolves : {wolf_wins:3d} wins  ({wolf_wins / games_to_run * 100:.1f}%)")

    # # Append results to the summary file (so override and append both accumulate)
    # out_path = (Path(__file__).parent / "game_logs" / scenario / "results_summary.txt").resolve()

    # # Read existing content if appending, so we don't lose prior games' records
    # existing = ""
    # if mode == "append" and out_path.exists():
    #     existing = out_path.read_text().strip() + "\n\n"

    # new_block = "\n".join(
    #     [
    #         f"Run: games {start_from:03d}-{start_from + num_games - 1:03d}  |  mode={mode}  |  scenario={scenario}",
    #         f"Villager model : {VILLAGER_MODEL}",
    #         f"Wolf model     : {WOLF_MODEL}",
    #         f"Villager wins  : {villager_wins} / {games_to_run}  ({villager_wins / games_to_run * 100:.1f}%)",
    #         f"Wolf wins      : {wolf_wins} / {games_to_run}  ({wolf_wins / games_to_run * 100:.1f}%)",
    #         "",
    #         "Per-game results:",
    #     ]
    #     + [
    #         f"  {r['game_id']}: {r['winner']}  " + ", ".join(f"{n}={rv.value}" for n, rv in r["roles"].items())
    #         for r in results
    #     ]
    # )

    # write_to_file(out_path, existing + new_block)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    args = parse_args()
    print(args)
    run(scenario=args.scenario, num_games=args.games, mode=args.mode)
