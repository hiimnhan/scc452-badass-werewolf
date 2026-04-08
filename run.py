import argparse

from config import MODEL_PROVIDERS
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

    players = ["Alice", "Bob", "Selena", "Raj", "Frank", "Joy", "Cyrus", "Emma", "Liam"]
    roles = {
        "Alice": "Doctor",
        "Bob": "Werewolf",
        "Selena": "Seer",
        "Raj": "Villager",
        "Frank": "Villager",
        "Joy": "Werewolf",
        "Cyrus": "Villager",
        "Emma": "Villager",
        "Liam": "Witch",
    }

    # TODO: implement the rest of the function
    pass


if __name__ == "__main__":
    args = parse_args()
    run(model_name=args.model_name)
