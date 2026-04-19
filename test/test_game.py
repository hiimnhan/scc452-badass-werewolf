from __future__ import annotations
from config import VILLAGER_MODEL, WOLF_MODEL
from players.guard import Guard
from players.seer import Seer
from players.witch import Witch
from players.villager import Villager
from players.wolf import Wolf
from players.coach import Coach
from dotenv import load_dotenv

from utils import get_llm

load_dotenv()
from players.base_player import Role
from run import build_player_objects
from game import GameState, Phase
from langchain_google_genai import ChatGoogleGenerativeAI
import os


VILLAGER_PLAYERS: list[str] = ["Alice", "Bob", "Selena", "Raj", "Frank"]
WOLF_PLAYERS: list[str] = ["Joy", "Cyrus"]
PLAYERS: list[str] = VILLAGER_PLAYERS + WOLF_PLAYERS

# Villager role pool — reshuffled randomly before every game.
# Must equal len(VILLAGER_PLAYERS).
VILLAGER_ROLE_POOL: list[Role] = [Role.SEER] * 1 + [Role.GUARD] * 1 + [Role.WITCH] * 1 + [Role.VILLAGER] * 2

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

roles = dict(zip(PLAYERS, VILLAGER_ROLE_POOL + ([Role.WEREWOLF] * len(WOLF_PLAYERS))))
villager_llm = get_llm(VILLAGER_MODEL)
wolf_llm = get_llm(WOLF_MODEL)

# villager_llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash", temperature=0.7, google_api_key=os.environ["GEMINI_API_KEY"]
# )
# wolf_llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash", temperature=0.7, google_api_key=os.environ["GEMINI_API_KEY"]
# )

game_id = "023"
scenario = "coach_and_self_analyze"

player_objects = build_player_objects(roles, villager_llm, wolf_llm, game_id, scenario)

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

# Shared config required by your GameState nodes[cite: 7]
config = {
    "configurable": {
        "player_objects": player_objects,
        "coach": coach,
        "scenario": scenario,
        "game_id": game_id,
        "MAX_DEBATE_TURNS": MAX_DEBATE_TURNS,
    }
}

# ============================================
# Run test: /opt/miniconda3/envs/scc452-badass-werewolf/bin/python -m test.test_game
# ============================================

if __name__ == "__main__":

    def test_wolf_debate():
        print("\n--- Testing WOLF DEBATE ---")
        state = initial_state.wolf_debate_node(initial_state, config)
        assert state._phase == Phase.ELIMINATE, f"Expected Phase.ELIMINATE, got {state._phase}"
        assert len(state._wolf_debate_log[state._round_num]) > 0, "Wolves did not log any debate statements."

    def test_eliminate():
        print("\n--- Testing ELIMINATE ---")
        state = initial_state.eliminate_node(initial_state, config)
        assert state._phase == Phase.PROTECT, f"Expected Phase.PROTECT, got {state._phase}"
        assert state._eliminated is not None, "Wolves failed to select a target."
        assert state._eliminated in state._alive_players, "Wolves selected an invalid target."

    def test_protect():
        print("\n--- Testing PROTECT ---")
        state = initial_state.protect_node(initial_state, config)
        assert state._phase == Phase.UNMASK, f"Expected Phase.UNMASK, got {state._phase}"
        assert state._protected is not None, "Guard failed to protect anyone."

    def test_unmask():
        print("\n--- Testing UNMASK ---")
        state = initial_state.unmask_node(initial_state, config)
        assert state._phase == Phase.SAVE_OR_POISON, f"Expected Phase.SAVE_OR_POISON, got {state._phase}"

    def test_save_or_poison():
        print("\n--- Testing SAVE OR POISON ---")
        state = initial_state.save_or_poison_node(initial_state, config)
        assert state._phase == Phase.RESOLVE_NIGHT, f"Expected Phase.RESOLVE_NIGHT, got {state._phase}"

    def test_resolve_night():
        print("\n--- Testing RESOLVE NIGHT ---")
        state = initial_state.resolve_night_node(initial_state, config)
        assert state._phase == Phase.CHECK_WINNER_NIGHT, f"Expected Phase.CHECK_WINNER_NIGHT, got {state._phase}"

    def test_check_winner_night():
        print("\n--- Testing CHECK WINNER NIGHT ---")
        state = initial_state.check_winner_night_node(initial_state, config)
        assert state._phase in [Phase.DEBATE, Phase.END], f"Expected Phase.DEBATE or Phase.END, got {state._phase}"

    def test_debate():
        print("\n--- Testing DEBATE ---")
        initial_state._phase = Phase.DEBATE  # Force state correctly
        initial_state._step = 0
        state = initial_state.debate_node(initial_state, config)
        assert state._phase in [Phase.DEBATE, Phase.VOTE], f"Expected Phase.DEBATE or VOTE, got {state._phase}"
        assert state._step == 1, "Debate step counter did not increment."

    def test_vote():
        print("\n--- Testing VOTE ---")
        state = initial_state.vote_node(initial_state, config)
        assert state._phase == Phase.EXILE, f"Expected Phase.EXILE, got {state._phase}"

    def test_exile():
        print("\n--- Testing EXILE ---")
        state = initial_state.exile_node(initial_state, config)
        assert state._phase == Phase.CHECK_WINNER_DAY, f"Expected Phase.CHECK_WINNER_DAY, got {state._phase}"

    def test_check_winner_day():
        print("\n--- Testing CHECK WINNER DAY ---")
        state = initial_state.check_winner_day_node(initial_state, config)
        assert state._phase in [Phase.WOLF_DEBATE, Phase.END], f"Expected Phase.WOLF_DEBATE or END, got {state._phase}"

    def test_end():
        print("\n--- Testing END NODE (Strategy & Compilations) ---")
        # Force a winner so the end node runs properly without crashing
        initial_state._winner = "Villagers"
        state = initial_state.end_node(initial_state, config)
        assert state is not None, "End node failed to return state."

    # ------------------------------------------
    # Dictionary mapping functions to status ("run" or "skip")
    test_dict = {
        test_wolf_debate: "skip",
        test_eliminate: "skip",
        test_protect: "skip",
        test_unmask: "skip",
        test_save_or_poison: "skip",
        test_resolve_night: "skip",
        test_check_winner_night: "skip",
        test_debate: "skip",
        test_vote: "skip",
        test_exile: "skip",
        test_check_winner_day: "skip",
        test_end: "skip",
    }

    print("\nStarting Test Pipeline...")
    for i, (test_func, info) in enumerate(test_dict.items()):
        func_name = test_func.__name__

        if info == "skip":
            print(f"⏭️ Test {i + 1} - function {func_name}: Skipped")
            continue

        try:
            test_func()
            print(f"✅ Test {i + 1} - function {func_name}: Passed")
        except Exception as e:
            print(f"❌ Test {i + 1} - function {func_name}: Failed ({e})")
            # We break the loop on failure because subsequent nodes rely on the state of previous nodes
            print("Stopping execution due to pipeline failure.")
            break
