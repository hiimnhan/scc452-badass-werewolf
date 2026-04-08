import argparse
from game import GameState
from config import MODEL_PROVIDERS
from players.guard import Guard
from utils import get_llm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Majority Influence Debate Simulation")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt-4o",
        choices=MODEL_PROVIDERS.keys(),
        help="Debate scenario to run (default: 1v1), choices: "
        + ", ".join(MODEL_PROVIDERS.keys()),
    )
    return parser.parse_args()


def run(model_name="gpt-4o"):
    llm = get_llm(model_name)

    players = ["Alice", "Bob", "Selena", "Raj", "Frank", "Joy", "Cyrus"]
    roles = {
        "Alice": "Guard",
        "Bob": "Seer", 
        "Selena": "Witch", 
        "Raj": "Villager", 
        "Frank": "Villager", 
        "Joy": "Werewolf", 
        "Cyrus": "Werewolf",
    }

    seer = next((p for p in players if roles[p] == "Seer"), None)
    guard = next((p for p in players if roles[p] == "Guard"), None)
    witch = next((p for p in players if roles[p] == "Witch"), None)
    villagers = [p for p in players if roles[p] == "Villager"]
    werewolves = [p for p in players if roles[p] == "Werewolf"]
    
    role_to_player_class = {
        "Guard": Guard,
        # TODO: replace these fallbacks when concrete role classes are implemented.
        "Seer": Guard,
        "Witch": Guard,
        "Villager": Guard,
        "Werewolf": Guard,
    }

    player_objects = {
        name: role_to_player_class[roles[name]](name=name, model=llm)
        for name in players
    }
    initial_state = GameState(
        round_num=0,
        players=players,
        alive_players=players.copy(),
        villagers=villagers,
        werewolves=werewolves,
        seer=seer,
        guard=guard,
        witch=witch,
        roles=roles
    )
    
    runnable = initial_state.build_graph()
    final_state = runnable.invoke(initial_state, config={
        "recursion_limit": 1000,
        "configurable": {
            "player_objects": player_objects,
            "MAX_DEBATE_TURNS": 6
        }
    })


if __name__ == "__main__":
    args = parse_args()
    run(model_name=args.model)
