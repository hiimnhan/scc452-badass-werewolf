from config import SCENARIO_CONFIG
from constants import FIGURES_FOLDERNAME, RESULTS_SUMMARY_FILENAME
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np

# /opt/miniconda3/envs/scc452-badass-werewolf/bin/python -m evaluation.learning_curve

# Learning-over-time trajectories. Unlike the per-bin bar charts, this plots a
# rolling-window estimate of each metric against the game index, so that any
# trend across the 100-game run (i.e. whether agents IMPROVE as coaching
# accumulates) is visible. Reads results_summary.csv, which already contains the
# per-game raw counts after win_rate.py, belief_calibration.py, and
# vote_precision.py have been run.

# Rolling-window length (games). Each plotted point aggregates the WINDOW games
# ending at that index. MIN_PERIODS controls how early the curve starts; we
# require a few games so the start is not single-game noise. Move these to
# constants.py if you prefer a project-wide setting.
WINDOW = 20
MIN_PERIODS = 10

# Locked color palette so a scenario keeps the same colour across every figure.
if isinstance(SCENARIO_CONFIG, dict):
    ALL_SCENARIOS = list(SCENARIO_CONFIG.keys())
else:
    ALL_SCENARIOS = list(SCENARIO_CONFIG)
_COLORS = sns.color_palette('Set2', len(ALL_SCENARIOS))
LOCKED_PALETTE = {scn: c for scn, c in zip(ALL_SCENARIOS, _COLORS)}

# Which scenarios make up each model-size comparison panel.
MODEL_SIZE_GROUPS = {
    "12b": ["baseline_12b_31B", "coach_12b_31B"],
    "31b": ["baseline_31B_31B", "coach_31B_31B"],
}

# (metric_key, y-axis label, y-limits)
METRICS = [
    ("win_rate",       "Villager Win Rate (%)", (0, 100)),
    ("belief_error",   "Belief RMSE",           (0, 1)),
    ("vote_precision", "Vote Precision (%)",    (0, 100)),
]


def load_scenario_csv(scenario):
    """Load a scenario's per-game results, sorted by game index. None if absent."""
    file_path = (Path(__file__).parent.parent / "game_logs" / scenario
                 / RESULTS_SUMMARY_FILENAME).resolve()
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return None
    df['game_id'] = df['game_id'].astype(int)
    return df.sort_values('game_id').reset_index(drop=True)


def rolling_trajectory(df, metric):
    """Return (game_index, rolling_value) arrays for one metric.

    Win Rate is a rolling mean of the binary villager-win indicator. Belief RMSE
    and Vote Precision are POOLED over the window (sum of numerators over sum of
    denominators) rather than a mean of per-game ratios, matching how the
    aggregate metrics are computed elsewhere.
    """
    g = df['game_id'].values

    if metric == "win_rate":
        win = (df['winner'] == 'Villagers').astype(float)
        series = win.rolling(WINDOW, min_periods=MIN_PERIODS).mean() * 100.0

    elif metric == "belief_error":
        num = df['belief_total_error'].rolling(WINDOW, min_periods=MIN_PERIODS).sum()
        den = df['belief_total_comparisons'].rolling(WINDOW, min_periods=MIN_PERIODS).sum()
        series = np.sqrt(num.where(den > 0) / den.where(den > 0))

    elif metric == "vote_precision":
        num = df['votes_against_wolf'].rolling(WINDOW, min_periods=MIN_PERIODS).sum()
        den = df['total_villager_votes'].rolling(WINDOW, min_periods=MIN_PERIODS).sum()
        series = (num.where(den > 0) / den.where(den > 0)) * 100.0

    else:
        raise ValueError(f"Unknown metric: {metric}")

    return g, series.values


def make_panel(group_label, scenarios, filename):
    """One figure, three side-by-side panels (win rate, belief RMSE, vote precision)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (metric, ylabel, ylim) in zip(axes, METRICS):
        for scenario in scenarios:
            df = load_scenario_csv(scenario)
            if df is None:
                continue
            x, y = rolling_trajectory(df, metric)
            ax.plot(x, y, label=scenario, color=LOCKED_PALETTE.get(scenario),
                    linewidth=2.2)
        ax.set_xlabel('Game Index', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_ylim(*ylim)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(title='Scenario', loc='best', fontsize=9)

    fig.suptitle(f"Learning Trajectories ({WINDOW}-game rolling window) — {group_label}",
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir = (Path(__file__).parent.parent / FIGURES_FOLDERNAME).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / filename
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Success! Chart saved to: {output_file}")


if __name__ == "__main__":
    # One combined panel per model size (ideal for the paper), plus an
    # all-scenarios panel for completeness.
    for size_label, scenarios in MODEL_SIZE_GROUPS.items():
        make_panel(size_label.upper(), scenarios, f"learning_curves_{size_label}.png")

    make_panel("All scenarios", ALL_SCENARIOS, "learning_curves_all.png")