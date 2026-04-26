from __future__ import annotations

import argparse
import re
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


GAME_LINE_RE = re.compile(r"^--- Game\s+(\d+)\s*/\s*(\d+)\s+\[(\d+)\]\s+---\s*$")
GAME_FOLDER_RE = re.compile(r"^game_(\d+)$")


@dataclass
class RunState:
    last_game: int | None = None
    last_total: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch a Werewolf run and restart it if it crashes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scenario", default="baseline", help="Scenario passed to run.py")
    parser.add_argument("--mode", default="override", choices=["append", "override"], help="Run mode")
    parser.add_argument(
        "--games",
        type=int,
        default=default_games_from_baseline(),
        help="Initial game count passed to run.py",
    )
    parser.add_argument(
        "--restart-threshold",
        type=int,
        default=20,
        help="If the most recent game number is below this value, derive --games from the console output when restarting.",
    )
    return parser.parse_args()


def default_games_from_baseline(target_total: int = 20) -> int:
    baseline_root = Path(__file__).parent / "game_logs" / "baseline"
    if not baseline_root.exists():
        return target_total

    existing_games = 0
    for folder in baseline_root.iterdir():
        if not folder.is_dir():
            continue
        if GAME_FOLDER_RE.match(folder.name):
            existing_games += 1

    return max(target_total - existing_games, 1) + 1


def build_command(games: int, scenario: str, mode: str) -> list[str]:
    return [sys.executable, "-u", "run.py", "--scenario", scenario, "--mode", mode, "--games", str(games)]


def stream_run(command: list[str]) -> tuple[int, RunState]:
    state = RunState()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        cwd=Path(__file__).parent,
        env=env,
    )

    assert process.stdout is not None

    for line in process.stdout:
        print(line, end="")
        match = GAME_LINE_RE.match(line.strip())
        if match:
            state.last_game = int(match.group(1))
            state.last_total = int(match.group(2))

    return_code = process.wait()
    return return_code, state


def main() -> int:
    args = parse_args()
    games = args.games
    attempt = 1

    while True:
        command = build_command(games, args.scenario, args.mode)
        print(f"\n[watch_run] attempt {attempt}: {' '.join(command)}\n")
        return_code, state = stream_run(command)

        if return_code == 0:
            print("\n[watch_run] run finished successfully.")
            return 0

        print(f"\n[watch_run] run exited with code {return_code}; restarting...\n")

        if state.last_game is not None and state.last_total is not None and state.last_game < args.restart_threshold:
            games = max(state.last_total - state.last_game + 1, 1)
            print(f"[watch_run] last seen game was {state.last_game} / {state.last_total}; restarting with --games {games}.")
        else:
            print(f"[watch_run] restarting with the original --games {games}.")

        attempt += 1


if __name__ == "__main__":
    raise SystemExit(main())