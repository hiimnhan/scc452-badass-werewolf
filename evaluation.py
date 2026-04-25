"""
analyze_metrics.py - Post-hoc analysis of Werewolf coaching experiment logs.

Computes the metric families specified in the research design:
  Family A — Belief accuracy (WIA, Brier, suspicion convergence)
  Family B — Decision quality (vote precision, power-role action precision)
  Family E — Cross-cutting (effectiveness, wolf-side comparator)
  DCR     — Directive Compliance Rate (from directives_uptake.json)

Inputs (per game folder under game_logs/<scenario>/game_NNN/):
  game_summary.json         public events + role assignments + winner
  suspicion_log.json        per-round suspicion dicts for every alive player
  directives_uptake.json    per-directive verdicts (absent in game 001)
  player_night_action_log/  seer/witch/guard private logs (used for B3)

Outputs (under game_logs/<scenario>/analysis/):
  per_game_metrics.csv      long-form: one row per (game, role, metric)
  summary_by_role.csv       aggregate stats per role
  plots/*.png               trajectory plots per metric per role

Usage:
  python analyze_metrics.py --scenario coach_and_self_analyze
  python analyze_metrics.py --scenario baseline --window 10
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Loaders
# ============================================================


def load_game(game_dir: Path) -> dict[str, Any] | None:
    """Load all JSON artefacts for one game. Returns None if game_summary missing."""
    summary_path = game_dir / "game_summary.json"
    if not summary_path.exists():
        return None

    with summary_path.open() as f:
        summary = json.load(f)

    suspicion_path = game_dir / "suspicion_log.json"
    suspicion = json.loads(suspicion_path.read_text()) if suspicion_path.exists() else {}

    uptake_path = game_dir / "directives_uptake.json"
    uptake = json.loads(uptake_path.read_text()) if uptake_path.exists() else None

    # Parse seer/witch/guard private logs for power-role precision.
    # These are markdown so we extract the structured bits with regex.
    night_dir = game_dir / "player_night_action_log"
    seer_actions = parse_seer_log(night_dir / "seer_log.md") if night_dir.exists() else []
    witch_actions = parse_witch_log(night_dir / "witch_log.md") if night_dir.exists() else []
    guard_actions = parse_guard_log(night_dir / "guard_log.md") if night_dir.exists() else []
    wolf_targets = parse_wolf_target_log(night_dir / "wolf_target_log.md") if night_dir.exists() else {}

    game_id = game_dir.name.replace("game_", "")
    return {
        "game_id": game_id,
        "summary": summary,
        "suspicion": suspicion,
        "uptake": uptake,
        "seer_actions": seer_actions,
        "witch_actions": witch_actions,
        "guard_actions": guard_actions,
        "wolf_targets": wolf_targets,
    }


def parse_seer_log(path: Path) -> list[dict]:
    """Returns [{round, target, is_wolf}] from seer_log.md."""
    if not path.exists():
        return []
    text = path.read_text()
    # Format: **Round N:** Investigated **TARGET** (Result: WOLF or Villager)
    pattern = re.compile(
        r"\*\*Round (\d+):\*\* Investigated \*\*([^*]+)\*\* \(Result: (WOLF|Villager)\)"
    )
    return [
        {"round": int(m.group(1)), "target": m.group(2).strip(), "is_wolf": m.group(3) == "WOLF"}
        for m in pattern.finditer(text)
    ]


def parse_witch_log(path: Path) -> list[dict]:
    """Returns [{round, saved, poisoned}] from witch_log.md."""
    if not path.exists():
        return []
    text = path.read_text()
    pattern = re.compile(r"\*\*Round (\d+):\*\* Saved \*\*([^*]+)\*\* \| Poisoned \*\*([^*]+)\*\*")
    out = []
    for m in pattern.finditer(text):
        saved = m.group(2).strip()
        poisoned = m.group(3).strip()
        out.append({
            "round": int(m.group(1)),
            "saved": None if saved == "None" else saved,
            "poisoned": None if poisoned == "None" else poisoned,
        })
    return out


def parse_guard_log(path: Path) -> list[dict]:
    """Returns [{round, target}] from guard_log.md."""
    if not path.exists():
        return []
    text = path.read_text()
    pattern = re.compile(r"\*\*Round (\d+):\*\* Protected \*\*([^*]+)\*\*")
    return [
        {"round": int(m.group(1)), "target": m.group(2).strip()}
        for m in pattern.finditer(text)
    ]


def parse_wolf_target_log(path: Path) -> dict[int, str]:
    """Returns {round: target_name}. Used to compute Guard Protection Accuracy."""
    if not path.exists():
        return {}
    text = path.read_text()
    pattern = re.compile(r"\*\*Round (\d+):\*\* Target \*\*([^*]+)\*\*")
    return {int(m.group(1)): m.group(2).strip() for m in pattern.finditer(text)}


# ============================================================
# Helpers
# ============================================================


def get_roles(game: dict) -> dict[str, str]:
    """Returns {player_name: role_string}. Handles both top-level and round-keyed shapes."""
    summary = game["summary"]
    roles = summary.get("roles", {})
    # roles dict has player names as keys, role strings as values
    return {k: v for k, v in roles.items() if isinstance(v, str)}


def get_wolves(game: dict) -> set[str]:
    return {p for p, r in get_roles(game).items() if r == "Werewolf"}


def get_villager_side(game: dict) -> set[str]:
    return {p for p, r in get_roles(game).items() if r != "Werewolf"}


def get_role_to_player(game: dict) -> dict[str, str]:
    """Returns {role: player_name}. For Werewolf there are multiple — returns first; use get_wolves() instead."""
    out = {}
    for p, r in get_roles(game).items():
        if r != "Werewolf":
            out[r] = p
    return out


# ============================================================
# Family A — Belief accuracy
# ============================================================


def compute_wia(game: dict) -> dict[str, float]:
    """Wolf Identification Accuracy per villager-side player (averaged across rounds).

    For each round, rank-percentile of the alive wolves in the player's
    suspicion ordering. 1.0 = wolves at top; 0.5 = random; 0.0 = at bottom.
    """
    suspicion = game["suspicion"]
    if not suspicion:
        return {}

    wolves = get_wolves(game)
    villager_side = get_villager_side(game)

    per_player_wia: dict[str, list[float]] = defaultdict(list)

    for round_str, round_log in suspicion.items():
        for player, susp in round_log.items():
            if player not in villager_side:
                continue
            if not isinstance(susp, dict) or not susp:
                continue

            # Alive wolves in this player's suspicion dict
            alive_wolves_in_view = [w for w in wolves if w in susp]
            if not alive_wolves_in_view:
                continue  # no alive wolves to rank — skip round

            # Rank all targets by score descending
            sorted_targets = sorted(
                susp.items(),
                key=lambda x: x[1].get("score", 0.5) if isinstance(x[1], dict) else 0.5,
                reverse=True,
            )
            n = len(sorted_targets)
            if n <= 1:
                continue

            # Average rank percentile of wolves (1.0 = first, 0.0 = last)
            wolf_percentiles = []
            for idx, (target, _) in enumerate(sorted_targets):
                if target in alive_wolves_in_view:
                    # Higher rank = better (lower index)
                    percentile = 1.0 - (idx / (n - 1))
                    wolf_percentiles.append(percentile)
            if wolf_percentiles:
                per_player_wia[player].append(sum(wolf_percentiles) / len(wolf_percentiles))

    return {p: sum(v) / len(v) for p, v in per_player_wia.items() if v}


def compute_brier(game: dict) -> dict[str, float]:
    """Brier score per villager-side player using final-round suspicions.

    Lower = better calibration. Uses the LAST round's suspicion log per player.
    """
    suspicion = game["suspicion"]
    if not suspicion:
        return {}

    wolves = get_wolves(game)
    villager_side = get_villager_side(game)

    # Aggregate per-player suspicions from the last round they appeared in
    last_round_per_player: dict[str, dict] = {}
    sorted_rounds = sorted(suspicion.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
    for round_str in sorted_rounds:
        for player, susp in suspicion[round_str].items():
            if player in villager_side and isinstance(susp, dict):
                last_round_per_player[player] = susp

    out = {}
    for player, susp in last_round_per_player.items():
        squared_errors = []
        for target, data in susp.items():
            if not isinstance(data, dict):
                continue
            score = data.get("score", 0.5)
            try:
                score = float(score)
            except (ValueError, TypeError):
                continue
            truth = 1.0 if target in wolves else 0.0
            squared_errors.append((score - truth) ** 2)
        if squared_errors:
            out[player] = sum(squared_errors) / len(squared_errors)
    return out


def compute_convergence_round(game: dict, threshold: float = 0.7) -> dict[str, int | None]:
    """First round in which a player's suspicion of any actual wolf reaches `threshold`.

    Returns None for that player if they never crossed the threshold.
    """
    suspicion = game["suspicion"]
    if not suspicion:
        return {}

    wolves = get_wolves(game)
    villager_side = get_villager_side(game)

    convergence: dict[str, int | None] = {p: None for p in villager_side}
    sorted_rounds = sorted(suspicion.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)

    for round_str in sorted_rounds:
        try:
            round_num = int(round_str)
        except (ValueError, TypeError):
            continue
        for player, susp in suspicion[round_str].items():
            if player not in villager_side or convergence.get(player) is not None:
                continue
            if not isinstance(susp, dict):
                continue
            for target, data in susp.items():
                if target not in wolves or not isinstance(data, dict):
                    continue
                try:
                    score = float(data.get("score", 0.0))
                except (ValueError, TypeError):
                    continue
                if score >= threshold:
                    convergence[player] = round_num
                    break

    return convergence


# ============================================================
# Family B — Decision quality
# ============================================================


def compute_vote_precision(game: dict) -> dict[str, float]:
    """Per-player fraction of votes cast against actual wolves.

    Only counts non-abstaining votes by villager-side voters.
    """
    summary = game["summary"]
    wolves = get_wolves(game)
    villager_side = get_villager_side(game)

    correct = defaultdict(int)
    total = defaultdict(int)

    for round_key, round_data in summary.items():
        if not isinstance(round_data, dict):
            continue
        votes = round_data.get("votes", [])
        if not isinstance(votes, list):
            continue
        for v in votes:
            voter = v.get("voter")
            target = v.get("vote_for")
            if voter not in villager_side or not target or str(target).lower() == "none":
                continue
            total[voter] += 1
            if target in wolves:
                correct[voter] += 1

    return {p: correct[p] / total[p] for p in total if total[p] > 0}


def compute_wolf_vote_share(game: dict) -> dict[int, float]:
    """Per-round fraction of villager votes landing on wolves. Keyed by round number."""
    summary = game["summary"]
    wolves = get_wolves(game)
    villager_side = get_villager_side(game)

    out = {}
    for round_key, round_data in summary.items():
        try:
            round_num = int(round_key)
        except (ValueError, TypeError):
            continue
        if not isinstance(round_data, dict):
            continue
        votes = round_data.get("votes", [])
        if not isinstance(votes, list):
            continue
        v_votes = [v for v in votes if v.get("voter") in villager_side and v.get("vote_for")]
        if not v_votes:
            continue
        wolf_hits = sum(1 for v in v_votes if v.get("vote_for") in wolves)
        out[round_num] = wolf_hits / len(v_votes)
    return out


def compute_seer_precision(game: dict) -> float | None:
    """Fraction of investigations landing on actual wolves."""
    actions = game["seer_actions"]
    if not actions:
        return None
    wolves = get_wolves(game)
    hits = sum(1 for a in actions if a["target"] in wolves)
    return hits / len(actions)


def compute_witch_poison_precision(game: dict) -> float | None:
    """Fraction of poisons landing on wolves."""
    poisons = [a["poisoned"] for a in game["witch_actions"] if a["poisoned"]]
    if not poisons:
        return None
    wolves = get_wolves(game)
    return sum(1 for t in poisons if t in wolves) / len(poisons)


def compute_witch_save_precision(game: dict) -> float | None:
    """Fraction of saves used on actual villagers (non-wolves)."""
    saves = [a["saved"] for a in game["witch_actions"] if a["saved"]]
    if not saves:
        return None
    wolves = get_wolves(game)
    return sum(1 for t in saves if t not in wolves) / len(saves)


def compute_guard_protection_accuracy(game: dict) -> float | None:
    """Fraction of nights the Guard's target matched the wolves' target."""
    guard_actions = game["guard_actions"]
    wolf_targets = game["wolf_targets"]
    if not guard_actions or not wolf_targets:
        return None
    matches = 0
    total = 0
    for action in guard_actions:
        wolf_target = wolf_targets.get(action["round"])
        if wolf_target is None:
            continue
        total += 1
        if action["target"] == wolf_target:
            matches += 1
    return matches / total if total > 0 else None


# ============================================================
# Family E — Cross-cutting
# ============================================================


def compute_role_effectiveness(game: dict) -> dict[str, float]:
    """For power roles: (rounds_acted * action_precision) / total_rounds.

    Captures both survival and quality of contribution.
    """
    summary = game["summary"]
    # Total rounds played in the game = max integer round key
    total_rounds = max(
        (int(k) for k in summary.keys() if str(k).isdigit()),
        default=1,
    )

    out = {}
    role_to_player = get_role_to_player(game)

    if "Seer" in role_to_player:
        sp = compute_seer_precision(game)
        if sp is not None:
            out[role_to_player["Seer"]] = (len(game["seer_actions"]) * sp) / total_rounds
    if "Guard" in role_to_player:
        gpa = compute_guard_protection_accuracy(game)
        if gpa is not None:
            out[role_to_player["Guard"]] = (len(game["guard_actions"]) * gpa) / total_rounds
    if "Witch" in role_to_player:
        pp = compute_witch_poison_precision(game)
        sp = compute_witch_save_precision(game)
        # Witch effectiveness combines both potions; use the mean of whichever exist
        parts = [x for x in [pp, sp] if x is not None]
        if parts:
            n_actions = len([a for a in game["witch_actions"] if a["saved"] or a["poisoned"]])
            out[role_to_player["Witch"]] = (n_actions * sum(parts) / len(parts)) / total_rounds
    return out


def compute_wolf_vote_precision(game: dict) -> dict[str, float]:
    """Wolf-side comparator: fraction of wolf votes against villagers."""
    summary = game["summary"]
    wolves = get_wolves(game)
    villager_side = get_villager_side(game)

    correct = defaultdict(int)
    total = defaultdict(int)

    for round_key, round_data in summary.items():
        if not isinstance(round_data, dict):
            continue
        votes = round_data.get("votes", [])
        if not isinstance(votes, list):
            continue
        for v in votes:
            voter = v.get("voter")
            target = v.get("vote_for")
            if voter not in wolves or not target or str(target).lower() == "none":
                continue
            total[voter] += 1
            if target in villager_side:
                correct[voter] += 1
    return {p: correct[p] / total[p] for p in total if total[p] > 0}


# ============================================================
# DCR (from directives_uptake.json)
# ============================================================


def compute_dcr(game: dict) -> dict[str, dict]:
    """Per-role DCR. Returns {role: {dcr, n_followed, n_ignored, n_partial, n_na}}."""
    uptake = game["uptake"]
    if not uptake:
        return {}

    by_role = uptake.get("by_role", {})
    out = {}
    for role, evals in by_role.items():
        if not isinstance(evals, list):
            continue
        followed = sum(1 for e in evals if e.get("verdict") == "FOLLOWED")
        ignored = sum(1 for e in evals if e.get("verdict") == "IGNORED")
        partial = sum(1 for e in evals if e.get("verdict") == "PARTIALLY_FOLLOWED")
        na = sum(1 for e in evals if e.get("verdict") == "NOT_APPLICABLE")
        applicable = followed + ignored + partial
        dcr = followed / applicable if applicable > 0 else None
        out[role] = {
            "dcr": dcr,
            "n_followed": followed,
            "n_ignored": ignored,
            "n_partial": partial,
            "n_na": na,
        }
    return out


# ============================================================
# Aggregation into long-form rows
# ============================================================


def build_rows(game: dict) -> list[dict]:
    """Flatten all metrics for one game into long-form rows."""
    rows = []
    game_id = game["game_id"]
    roles = get_roles(game)
    winner = game["summary"].get("winner")

    def add(role: str, player: str, metric: str, value):
        if value is None:
            return
        rows.append({
            "game_id": game_id,
            "role": role,
            "player": player,
            "metric": metric,
            "value": value,
            "winner": winner,
        })

    # Family A
    for player, wia in compute_wia(game).items():
        add(roles.get(player, "Unknown"), player, "WIA", wia)
    for player, brier in compute_brier(game).items():
        add(roles.get(player, "Unknown"), player, "Brier", brier)
    for player, conv in compute_convergence_round(game).items():
        if conv is not None:
            add(roles.get(player, "Unknown"), player, "ConvergenceRound", conv)

    # Family B
    for player, vp in compute_vote_precision(game).items():
        add(roles.get(player, "Unknown"), player, "VotePrecision", vp)
    for round_num, share in compute_wolf_vote_share(game).items():
        rows.append({
            "game_id": game_id,
            "role": "AllVillagers",
            "player": f"round_{round_num}",
            "metric": "WolfVoteShare",
            "value": share,
            "winner": winner,
        })

    role_to_player = get_role_to_player(game)
    if "Seer" in role_to_player:
        sp = compute_seer_precision(game)
        if sp is not None:
            add("Seer", role_to_player["Seer"], "SeerInvestigationPrecision", sp)
    if "Guard" in role_to_player:
        gpa = compute_guard_protection_accuracy(game)
        if gpa is not None:
            add("Guard", role_to_player["Guard"], "GuardProtectionAccuracy", gpa)
    if "Witch" in role_to_player:
        pp = compute_witch_poison_precision(game)
        if pp is not None:
            add("Witch", role_to_player["Witch"], "WitchPoisonPrecision", pp)
        sp = compute_witch_save_precision(game)
        if sp is not None:
            add("Witch", role_to_player["Witch"], "WitchSavePrecision", sp)

    # Family E
    for player, eff in compute_role_effectiveness(game).items():
        add(roles.get(player, "Unknown"), player, "RoleEffectiveness", eff)
    for player, wvp in compute_wolf_vote_precision(game).items():
        add("Werewolf", player, "WolfVotePrecision", wvp)

    # DCR
    for role, dcr_data in compute_dcr(game).items():
        if dcr_data["dcr"] is not None:
            rows.append({
                "game_id": game_id,
                "role": role,
                "player": "_aggregate",
                "metric": "DCR",
                "value": dcr_data["dcr"],
                "winner": winner,
            })

    return rows


# ============================================================
# Plotting
# ============================================================


def plot_trajectory(df: pd.DataFrame, metric: str, out_path: Path, window: int = 10):
    """Plot per-role trajectory of `metric` over game_id, with rolling-window smoothing."""
    sub = df[df["metric"] == metric].copy()
    if sub.empty:
        return

    sub["game_num"] = sub["game_id"].astype(int)
    sub = sub.sort_values("game_num")

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = 0
    for role, group in sub.groupby("role"):
        per_game = group.groupby("game_num")["value"].mean().sort_index()
        if len(per_game) < 2:
            continue
        smoothed = per_game.rolling(window=window, min_periods=1).mean()
        ax.plot(per_game.index, smoothed.values, label=role, linewidth=2)
        ax.scatter(per_game.index, per_game.values, alpha=0.2, s=10)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    ax.set_title(f"{metric} trajectory (rolling window = {window} games)")
    ax.set_xlabel("Game number")
    ax.set_ylabel(metric)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, help="Scenario name (folder under game_logs/)")
    parser.add_argument("--logs-root", default="game_logs", help="Root logs directory")
    parser.add_argument("--window", type=int, default=10, help="Rolling-mean window for plots")
    args = parser.parse_args()

    base = Path(args.logs_root) / args.scenario
    if not base.exists():
        raise SystemExit(f"Scenario folder not found: {base}")

    out_dir = base / "analysis"
    out_dir.mkdir(exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Walk game folders in order
    game_dirs = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith("game_")],
        key=lambda d: int(d.name.split("_")[1]),
    )

    print(f"Found {len(game_dirs)} game folders under {base}")

    all_rows: list[dict] = []
    skipped = 0
    for game_dir in game_dirs:
        game = load_game(game_dir)
        if game is None:
            skipped += 1
            continue
        try:
            all_rows.extend(build_rows(game))
        except Exception as e:
            print(f"  ! failed to build rows for {game_dir.name}: {e}")
            skipped += 1

    if skipped:
        print(f"Skipped {skipped} games due to missing files or errors.")

    if not all_rows:
        raise SystemExit("No metric rows produced. Check scenario folder.")

    df = pd.DataFrame(all_rows)
    per_game_path = out_dir / "per_game_metrics.csv"
    df.to_csv(per_game_path, index=False)
    print(f"Wrote {len(df)} rows -> {per_game_path}")

    # Summary by role + metric
    summary = (
        df.groupby(["role", "metric"])["value"]
        .agg(["count", "mean", "std", "min", "max"])
        .round(4)
        .reset_index()
    )
    summary_path = out_dir / "summary_by_role.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote summary -> {summary_path}")

    # Trajectory plots
    metrics_to_plot = [
        "DCR", "WIA", "Brier", "ConvergenceRound",
        "VotePrecision", "SeerInvestigationPrecision",
        "GuardProtectionAccuracy", "WitchPoisonPrecision", "WitchSavePrecision",
        "RoleEffectiveness", "WolfVotePrecision",
    ]
    for metric in metrics_to_plot:
        out_path = plot_dir / f"{metric}.png"
        plot_trajectory(df, metric, out_path, window=args.window)
    print(f"Wrote plots -> {plot_dir}")

    # Headline numbers
    print("\nHeadline metrics (overall mean across all games):")
    overall = df.groupby("metric")["value"].mean().round(4)
    for metric, value in overall.items():
        print(f"  {metric:<32} {value}")


if __name__ == "__main__":
    main()