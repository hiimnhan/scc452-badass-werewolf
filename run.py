"""
run.py — Werewolf experiment runner

Usage
-----
Single scenario, 100 games (default):
    python run.py --scenario baseline

Different scenario:
    python run.py --scenario coach_and_self_analyze

Override number of games:
    python run.py --scenario baseline --games 10

File layout produced
--------------------
game_logs/
  {scenario}/
    game_001/
      game_summary.txt        all public announcements in order
      debate_log.txt          every statement from every day debate
      vote_log.txt            every vote cast across all rounds
      roles_this_game.txt     who played which role
      {name}_{role}_note.txt  each player's compiled note (game summary + suspicions)
      coach_feedback.txt      coach's post-game feedback (if coaching enabled)

strategies/
  {scenario}/
    {name}_strategy.txt  each player's accumulated strategy (updated after every game)
    coach_strategy.txt          coach's accumulated coaching strategy
"""

import argparse
import random
from pathlib import Path

from game import GameState
from config import MODEL_PROVIDERS, SCENARIO_CONFIG, VILLAGER_MODEL, WOLF_MODEL
from players.guard import Guard
from players.seer import Seer
from players.witch import Witch
from players.base_player import BasePlayer
from players.wolf import Wolf
from players.coach import Coach
from players.base_player import Role, VILLAGER_SIDE, WOLF_SIDE
from utils import get_llm


# ============================================================
# Player roster & role pool
# ============================================================

VILLAGER_PLAYERS: list[str] = ["Alice", "Bob", "Selena", "Raj", "Frank"]
WOLF_PLAYERS: list[str] = ["Joy", "Cyrus"]
PLAYERS: list[str] = VILLAGER_PLAYERS + WOLF_PLAYERS

# Villager Role pool — reshuffled randomly before every game.
# Edit counts here to change game balance; must equal len(VILLAGER_PLAYERS).
VILLAGER_ROLE_POOL: list[Role] = (
    + [Role.SEER]     * 1
    + [Role.GUARD]    * 1
    + [Role.WITCH]    * 1
    + [Role.VILLAGER] * 2
)

assert len(VILLAGER_ROLE_POOL) == len(VILLAGER_PLAYERS), f"VILLAGER_ROLE_POOL has {len(VILLAGER_ROLE_POOL)} entries but VILLAGER_PLAYERS has {len(VILLAGER_PLAYERS)}."

ROLE_TO_CLASS: dict[Role, type] = {
    Role.VILLAGER: BasePlayer,
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
    parser.add_argument(
        "-s", "--scenario",
        type=str,
        default="baseline",
        choices=list(SCENARIO_CONFIG.keys()),
        help="Experiment scenario to run",
    )
    parser.add_argument(
        "-g", "--games",
        type=int,
        default=100,
        help="Number of games to run back-to-back",
    )
    return parser.parse_args()


# ============================================================
# Role assignment
# ============================================================

def assign_roles_randomly() -> dict[str, Role]:
    """Shuffle the role pool and assign one role to each player."""
    shuffled = VILLAGER_ROLE_POOL.copy()
    random.shuffle(shuffled)
    return dict(zip(PLAYERS, shuffled + ([Role.WEREWOLF] * 2)))


# ============================================================
# Player construction
# ============================================================

def build_player_objects(
    roles: dict[str, Role],
    villager_llm,
    wolf_llm,
    game_id: str,
    scenario: str,
) -> dict[str, object]:
    """Instantiate one player object per player for the given role assignment.

    Villager-side players get the small model; wolves get the large model.
    Player objects are created fresh every game because the role (and therefore the class) can change. Strategies persist on disk and are loaded at init.
    """
    player_objects: dict[str, BasePlayer] = {}
    for name in PLAYERS:
        role = roles[name]
        model = wolf_llm if role in WOLF_SIDE else villager_llm
        cls = ROLE_TO_CLASS[role]
        player_objects[name] = cls(
            name=name,
            model=model,
            game_id=game_id,
            scenario=scenario,
        )

    # Seed suspicion scores — pure Python, no LLM call needed
    for name, player_obj in player_objects.items():
        player_obj.init_suspicions([p for p in PLAYERS if p != name])

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
    """Build player objects, run one game, and return the final GameState."""

    player_objects = build_player_objects(roles, villager_llm, wolf_llm, game_id, scenario)
    coach = Coach(model=wolf_llm, game_id=game_id, scenario=scenario)

    # Derive GameState fields from the role dict
    seer       = next((p for p in PLAYERS if roles[p] == Role.SEER),  None)
    guard      = next((p for p in PLAYERS if roles[p] == Role.GUARD), None)
    witch      = next((p for p in PLAYERS if roles[p] == Role.WITCH), None)
    villagers  = [p for p in PLAYERS if roles[p] == Role.VILLAGER]
    werewolves = [p for p in PLAYERS if roles[p] == Role.WEREWOLF]

    initial_state = GameState(
        round_num=0,
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
        initial_state,
        config={
            "recursion_limit": 1000,
            "configurable": {
                "player_objects":    player_objects,
                "MAX_DEBATE_TURNS":  MAX_DEBATE_TURNS,
                "scenario":          scenario,
                "game_id":           game_id,   # used by end_node for file paths
                "coach":             coach,
            },
        },
    )
    return final_state


# ============================================================
# Multi-game experiment loop
# ============================================================

def run(scenario: str = "baseline", num_games: int = 100) -> None:
    """
    Run num_games games back-to-back under the given scenario.

    Roles are reshuffled randomly before every game.
    Villager-side players share a small model; wolf-side players share a large model.
    Strategies accumulate on disk across games — each player learns from all the
    games they play, indexed by their role in that game.
    """
    print(f"\n{'='*62}")
    print(f"  EXPERIMENT: {scenario}  |  {num_games} games")
    print(f"  Villager model : {VILLAGER_MODEL}")
    print(f"  Wolf model     : {WOLF_MODEL}")
    print(f"{'='*62}\n")

    # Build LLM instances once — reused across all games
    villager_llm = get_llm(VILLAGER_MODEL)
    wolf_llm     = get_llm(WOLF_MODEL)

    results: list[dict] = []

    for i in range(1, num_games + 1):
        game_id = f"{i:03d}"   # path becomes game_logs/{scenario}/game_{i:03d}/

        # Random role assignment for this game
        roles = assign_roles_randomly()

        # Print header
        role_summary = ", ".join(f"{n}={r.value}" for n, r in roles.items())
        print(f"\n--- Game {i:3d} / {num_games}  [{game_id}] ---")
        print(f"    Roles: {role_summary}")

        try:
            final_state = run_game(villager_llm, wolf_llm, roles, game_id, scenario)
            winner = final_state._winner or "Unknown"
        except Exception as e:
            # Log the failure and continue — don't let one bad game kill 100
            print(f"    !! Game {game_id} crashed: {e}")
            winner = "Error"

        results.append({"game_id": game_id, "winner": winner, "roles": roles})
        print(f"    Winner: {winner}")

    # ── Final summary ──────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  FINAL RESULTS — {scenario} ({num_games} games)")
    print(f"{'='*62}")

    villager_wins = sum(1 for r in results if r["winner"] == "Villagers")
    wolf_wins     = sum(1 for r in results if r["winner"] == "Werewolves")
    errors        = sum(1 for r in results if r["winner"] == "Error")

    for r in results:
        print(f"  {r['game_id']}: {r['winner']}")

    print(f"\n  Villagers  : {villager_wins:3d} wins  ({villager_wins/num_games*100:.1f}%)")
    print(f"  Werewolves : {wolf_wins:3d} wins  ({wolf_wins/num_games*100:.1f}%)")
    if errors:
        print(f"  Errors     : {errors:3d}")

    # Save results summary to disk
    from utils import write_to_file
    summary_lines = [
        f"Experiment : {scenario}",
        f"Games      : {num_games}",
        f"Villager model : {VILLAGER_MODEL}",
        f"Wolf model     : {WOLF_MODEL}",
        "",
        f"Villager wins : {villager_wins} ({villager_wins/num_games*100:.1f}%)",
        f"Wolf wins     : {wolf_wins} ({wolf_wins/num_games*100:.1f}%)",
        "",
        "Per-game results:",
    ] + [
        f"  {r['game_id']}: {r['winner']}  "
        + ", ".join(f"{n}={rv.value}" for n, rv in r["roles"].items())
        for r in results
    ]
    out_path = (
        Path(__file__).parent / "game_logs" / scenario / "results_summary.txt"
    ).resolve()
    write_to_file(out_path, "\n".join(summary_lines))
    print(f"\n  Results saved to: {out_path}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    args = parse_args()
    run(scenario=args.scenario, num_games=args.games)