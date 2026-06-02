import pandas as pd
import numpy as np
from config import SCENARIO_CONFIG
from pathlib import Path

# /opt/miniconda3/envs/scc452-badass-werewolf/bin/python -m evaluation.block_bootstrap

# Bootstrapping parameters
PRIMARY_BLOCK_SIZE = 10                 # block size used for the results reported in the paper
ROBUSTNESS_BLOCK_SIZES = [5, 10, 20]    # block sizes reported in the supplementary material
N_BOOTSTRAPS = 10000  # 10,000 iterations for a highly stable confidence interval

# Baseline -> coach pairs whose difference (coach - baseline) we test for significance.
DIFFERENCE_PAIRS = [
    ("baseline_12b_31B", "coach_12b_31B"),
    ("baseline_31B_31B", "coach_31B_31B"),
]

# Metrics for which a coach-minus-baseline difference is defined (Compliance and
# Summarisation have no baseline value, so no difference is computed for them).
# (key, label, decimal_places, is_proportion, diff_divisor)
#
# is_proportion flags metrics that are bounded rates in [0, 1] (Win Rate and Vote
# Precision). For these the proportion-appropriate effect size is Cohen's h
# (arcsine transform), not Cohen's d. Belief Error is a continuous RMSE, so it
# uses Cohen's d.
#
# diff_divisor rescales the reported difference so that both proportion metrics
# are expressed on the SAME [0, 1] scale. Win Rate is already a proportion
# (divisor 1). Vote Precision is computed as a percentage in compute_metrics()
# (so the per-scenario tables can display it as a %), so we divide its difference
# by 100 to report it as a proportion too (e.g. +0.16 rather than +16.2), keeping
# the difference table consistent with how Win Rate is written. Belief Error is
# reported in its native RMSE units (divisor 1).
DIFF_METRICS = [
    ("win_rate",       "Win Rate",       3, True,  1.0),
    ("belief_error",   "Belief Error",   3, False, 1.0),
    ("vote_precision", "Vote Precision", 3, True,  100.0),
]


# --- 1. Load the precalculated LLM Evaluation Data ---
try:
    llm_eval = pd.read_csv("evaluation/llm_eval_precalculated.csv")
except FileNotFoundError:
    print("Warning: llm_eval_precalculated.csv not found. LLM metrics will be skipped.")
    llm_eval = pd.DataFrame()


def load_scenario_df(scenario):
    """Load a scenario's per-game dataframe with LLM metrics merged in.

    Returns the dataframe, or None if the file is missing or required
    columns are absent (with an explanatory message printed).
    """
    file_path = Path("game_logs") / scenario / 'results_summary.csv'

    if not file_path.exists():
        print(f"\nSkipping {scenario}: File not found at {file_path}")
        return None

    df = pd.read_csv(file_path)

    # --- Merge LLM Metrics into the Game DataFrame ---
    if not llm_eval.empty:
        scenario_llm = llm_eval[llm_eval['scenario'] == scenario]
        # Aggregate the LLM metrics by game_id to match the 1-row-per-game structure
        llm_agg = scenario_llm.groupby('game_id').agg(
            follow_count=('follow_count', 'sum'),
            total_directives=('total_directives', 'sum'),
            summarization_score=('summarization_score', 'sum'),
            llm_row_count=('game_id', 'count')
        ).reset_index()
        df = df.merge(llm_agg, on='game_id', how='left')

    required_cols = ['winner', 'belief_total_error', 'belief_total_comparisons',
                     'total_villager_votes', 'votes_against_wolf']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"\nSkipping {scenario}: Missing columns {missing_cols}. "
              f"Please run belief and vote precision scripts first.")
        return None

    return df


def compute_metrics(boot_df):
    """Compute all five evaluation metrics on a (resampled) dataframe.

    Returns a dict. LLM metrics are None when no directives/rows are present
    (e.g. baseline scenarios with no coach), so they can be cleanly skipped.
    """
    # 1. Villager Win Rate
    win_rate = (boot_df['winner'] == 'Villagers').mean()

    # 2. Belief Error (RMSE)
    total_error = boot_df['belief_total_error'].sum()
    total_comps = boot_df['belief_total_comparisons'].sum()
    rmse = np.sqrt(total_error / total_comps) if total_comps > 0 else 0.0

    # 3. Vote Precision (%)
    total_votes = boot_df['total_villager_votes'].sum()
    wolf_votes = boot_df['votes_against_wolf'].sum()
    precision = (wolf_votes / total_votes * 100) if total_votes > 0 else 0.0

    # 4 & 5. LLM Metrics (None when undefined, e.g. no coach feedback)
    compliance = None
    summarization = None
    if 'follow_count' in boot_df.columns:
        tot_dir = boot_df['total_directives'].sum()
        if tot_dir > 0:
            compliance = boot_df['follow_count'].sum() / tot_dir * 100
        tot_rows = boot_df['llm_row_count'].sum()
        if tot_rows > 0 and boot_df['summarization_score'].sum() > 0:
            summarization = boot_df['summarization_score'].sum() / tot_rows

    return {
        'win_rate': win_rate,
        'belief_error': rmse,
        'vote_precision': precision,
        'compliance': compliance,
        'summarization': summarization,
    }


def moving_block_indices(n_games, block_size, rng):
    """Draw moving (overlapping) blocks reconstructing a run of length n_games.

    Start indices range over [0, n_games - block_size], so blocks may overlap.
    """
    n_blocks = n_games // block_size
    starts = rng.integers(0, n_games - block_size + 1, size=n_blocks)
    idx = []
    for start in starts:
        idx.extend(range(start, start + block_size))
    return idx


def bootstrap_scenario(df, block_size, n_boot, seed):
    """Moving block bootstrap for a single scenario at a given block size.

    Returns (point_estimates, bootstrap_distributions) where each is a dict
    keyed by metric name. The full per-replicate arrays are retained (not just
    summary CIs) so that the coach-minus-baseline difference can be formed from
    the SAME draws (see difference section below). LLM distributions are empty
    lists when undefined. Each scenario is given its own independent RNG stream
    via a distinct seed, so draws across scenarios are independent.
    """
    rng = np.random.default_rng(seed)
    n_games = len(df)

    dist = {k: [] for k in ['win_rate', 'belief_error', 'vote_precision',
                            'compliance', 'summarization']}

    for _ in range(n_boot):
        idx = moving_block_indices(n_games, block_size, rng)
        m = compute_metrics(df.iloc[idx])
        for k, v in m.items():
            if v is not None:
                dist[k].append(v)

    point = compute_metrics(df)
    return point, dist


def ci(values):
    """95% percentile CI; returns None if the distribution is empty."""
    if values is None or len(values) == 0:
        return None
    return np.percentile(values, [2.5, 97.5])


def fmt_ci(c, pct=False, dp=3):
    if c is None:
        return "--"
    if pct:
        return f"[{c[0]:.1f}%, {c[1]:.1f}%]"
    return f"[{c[0]:.{dp}f}, {c[1]:.{dp}f}]"


def scenario_seed(scenario):
    """Deterministic, distinct seed per scenario -> independent RNG streams.

    Distinct seeds guarantee that the resamples drawn for two scenarios are
    independent, which is what licenses forming the unpaired difference by
    subtracting their per-replicate metric arrays. We never share block indices
    across scenarios (which would imply a meaningless game-to-game pairing).

    Works whether SCENARIO_CONFIG is a list or a dict (uses key order).
    """
    scenarios = list(SCENARIO_CONFIG)  # for a dict this is the list of keys
    return 42 + 1000 * scenarios.index(scenario)


# ============================================================
# SINGLE BOOTSTRAP PASS (block size = 10), reported in the paper.
#
# One pass over the scenarios produces BOTH:
#   (A) the per-scenario 95% CIs (the main results table), and
#   (B) the coach - baseline difference CIs (the significance test),
# the latter formed by subtracting the SAME per-replicate draws, so the two
# analyses are mutually consistent. The output format of part (A) is preserved
# from the original script.
# ============================================================
print("\n" + "#" * 60)
print(f"# PRIMARY RESULTS (single pass, block size = {PRIMARY_BLOCK_SIZE})")
print("#" * 60)

# Store each scenario's point estimates and full per-replicate draws.
scenario_results = {}

# ---- (A) Per-scenario confidence intervals ----
for scenario in SCENARIO_CONFIG:
    df = load_scenario_df(scenario)
    if df is None:
        continue

    point, dist = bootstrap_scenario(
        df, PRIMARY_BLOCK_SIZE, N_BOOTSTRAPS, seed=scenario_seed(scenario)
    )
    scenario_results[scenario] = {'point': point, 'dist': dist}

    ci_win = ci(dist['win_rate'])
    ci_belief = ci(dist['belief_error'])
    ci_vote = ci(dist['vote_precision'])

    print(f"\n=== Block Bootstrapping Results: {scenario} ===")
    print(f"Block Size: {PRIMARY_BLOCK_SIZE} | Iterations: {N_BOOTSTRAPS}")
    print("-" * 50)
    print(f"1. Villager Win Rate:")
    print(f"   Original Data: {point['win_rate']:.3f}")
    print(f"   Boot Mean:     {np.mean(dist['win_rate']):.3f}")
    print(f"   95% CI:        [{ci_win[0]:.3f}, {ci_win[1]:.3f}]\n")

    print(f"2. Belief Error (RMSE):")
    print(f"   Original Data: {point['belief_error']:.3f}")
    print(f"   Boot Mean:     {np.mean(dist['belief_error']):.3f}")
    print(f"   95% CI:        [{ci_belief[0]:.3f}, {ci_belief[1]:.3f}]\n")

    print(f"3. Vote Precision (%):")
    print(f"   Original Data: {point['vote_precision']:.1f}%")
    print(f"   Boot Mean:     {np.mean(dist['vote_precision']):.1f}%")
    print(f"   95% CI:        [{ci_vote[0]:.1f}%, {ci_vote[1]:.1f}%]")

    if dist['compliance']:
        ci_comp = ci(dist['compliance'])
        print(f"\n4. Compliance Rate (%):")
        print(f"   Original Data: {point['compliance']:.1f}%")
        print(f"   Boot Mean:     {np.mean(dist['compliance']):.1f}%")
        print(f"   95% CI:        [{ci_comp[0]:.1f}%, {ci_comp[1]:.1f}%]")

        ci_summ = ci(dist['summarization'])
        print(f"\n5. Summarization Ability (0-10):")
        print(f"   Original Data: {point['summarization']:.2f}")
        print(f"   Boot Mean:     {np.mean(dist['summarization']):.2f}")
        print(f"   95% CI:        [{ci_summ[0]:.2f}, {ci_summ[1]:.2f}]")
    print("=" * 50)


# ---- (B) Difference CIs, reusing the SAME per-replicate draws ----
# For each baseline/coach pair we subtract the two scenarios' per-replicate
# metric arrays elementwise. Because the scenarios used independent RNG streams,
# these are independent draws, so the elementwise difference is the correct
# UNPAIRED difference distribution. The replicate index is bookkeeping only; it
# does NOT pair game i of one run with game i of the other (the runs are separate
# simulations with no game-to-game correspondence).
print("\n\n" + "#" * 60)
print(f"# COACHING EFFECT: DIFFERENCE CIs (coach - baseline, block size = {PRIMARY_BLOCK_SIZE})")
print("#  Derived from the same draws as the per-scenario CIs above.")
print("#" * 60)

for base_scn, coach_scn in DIFFERENCE_PAIRS:
    if base_scn not in scenario_results or coach_scn not in scenario_results:
        print(f"\nSkipping pair ({coach_scn} - {base_scn}): a run is missing.")
        continue

    base_dist = scenario_results[base_scn]['dist']
    coach_dist = scenario_results[coach_scn]['dist']
    base_pt = scenario_results[base_scn]['point']
    coach_pt = scenario_results[coach_scn]['point']

    print(f"\n=== {coach_scn}  minus  {base_scn} ===")
    print(f"Block Size: {PRIMARY_BLOCK_SIZE} | Iterations: {N_BOOTSTRAPS}")
    print("-" * 64)

    for key, label, dp, _is_prop, divisor in DIFF_METRICS:
        # Elementwise difference of independent per-replicate draws, rescaled by
        # divisor so proportion metrics are reported on a common [0, 1] scale.
        diffs = (np.array(coach_dist[key]) - np.array(base_dist[key])) / divisor
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        point_diff = (coach_pt[key] - base_pt[key]) / divisor
        excludes_zero = (lo > 0) or (hi < 0)
        verdict = "SIGNIFICANT (CI excludes 0)" if excludes_zero \
            else "not significant (CI includes 0)"
        print(f"{label}:")
        print(f"   Difference (coach - baseline): {point_diff:+.{dp}f}")
        print(f"   95% CI of difference:          [{lo:+.{dp}f}, {hi:+.{dp}f}]")
        print(f"   Verdict:                       {verdict}")
    print("  (Win Rate / Vote Precision: positive = improvement; "
          "Belief Error: negative = improvement.)")
    print("=" * 64)


# ============================================================
# BLOCK-SIZE ROBUSTNESS (5 / 10 / 20), supplementary material.
# Shows that the 95% CIs are stable across block-length choices,
# justifying the block size of 10 used in the paper. This is a
# separate sensitivity analysis and re-resamples at each block size.
# ============================================================
print("\n\n" + "#" * 60)
print(f"# BLOCK-SIZE ROBUSTNESS  (sizes = {ROBUSTNESS_BLOCK_SIZES})")
print("#  Reported in the Supplementary Material.")
print("#" * 60)

COL_W = 24

for scenario in SCENARIO_CONFIG:
    df = load_scenario_df(scenario)
    if df is None:
        continue

    results = {}
    for b in ROBUSTNESS_BLOCK_SIZES:
        point, dist = bootstrap_scenario(
            df, b, N_BOOTSTRAPS, seed=scenario_seed(scenario)
        )
        results[b] = {
            'point': point,
            'win': ci(dist['win_rate']),
            'belief': ci(dist['belief_error']),
            'vote': ci(dist['vote_precision']),
            'comp': ci(dist['compliance']),
            'summ': ci(dist['summarization']),
        }

    point = results[ROBUSTNESS_BLOCK_SIZES[0]]['point']  # point est. is block-size-independent

    print(f"\n=== Block-Size Robustness: {scenario} ===")
    print(f"Iterations per block size: {N_BOOTSTRAPS}")
    header = f"{'Metric (point est.)':<26}" + "".join(f"{'L=' + str(b):>{COL_W}}" for b in ROBUSTNESS_BLOCK_SIZES)
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    def print_row(label, point_val, key, pct=False, dp=3):
        lbl = f"{label} ({point_val})"
        cells = "".join(f"{fmt_ci(results[b][key], pct=pct, dp=dp):>{COL_W}}" for b in ROBUSTNESS_BLOCK_SIZES)
        print(f"{lbl:<26}{cells}")

    print_row("Win Rate",     f"{point['win_rate']:.3f}",      'win',    dp=3)
    print_row("Belief Error", f"{point['belief_error']:.3f}",  'belief', dp=3)
    print_row("Vote Prec.",   f"{point['vote_precision']:.1f}%", 'vote',  pct=True)
    if point['compliance'] is not None:
        print_row("Compliance", f"{point['compliance']:.1f}%",   'comp',  pct=True)
    if point['summarization'] is not None:
        print_row("Summ. Ability", f"{point['summarization']:.2f}", 'summ', dp=2)
    print("=" * len(header))


# ============================================================
# EFFECT SIZES (coach vs baseline), main paper.
#
# The difference bootstrap above establishes whether the coaching effect is
# significant; effect size reports how LARGE it is, independent of n. The choice
# of effect size follows the nature of each metric:
#   * Win Rate and Vote Precision are PROPORTIONS (bounded rates in [0, 1]), so
#     the proportion-appropriate effect size is Cohen's h, computed from the
#     aggregate proportions via the arcsine transform.
#   * Belief Error is a continuous RMSE, so we use Cohen's d (standardised mean
#     difference) on its per-game values.
#
# Note: d/h are descriptive magnitudes and do not assume independence; the
# significance of the effect is governed by the block bootstrap, not by these.
# Convention: positive = coach higher than baseline. For Belief Error a higher
# value is worse, so a positive d there denotes a larger (worse) error.
# ============================================================
def per_game_values(df, metric):
    """Per-game metric values as a 1-D array, dropping games where undefined.

    Used for Cohen's d (Belief Error). Proportion metrics use aggregate
    proportions with Cohen's h instead (see proportion() below).
    """
    if metric == "belief_error":
        sub = df[df['belief_total_comparisons'] > 0]
        return np.sqrt(sub['belief_total_error'] / sub['belief_total_comparisons']).values
    raise ValueError(f"per_game_values is for continuous metrics only, got {metric!r}")


def proportion(df, metric):
    """Aggregate proportion in [0, 1] for a proportion-valued metric.

    Win Rate is the fraction of games won; Vote Precision is the fraction of
    villager votes that hit a wolf, pooled over all games.
    """
    if metric == "win_rate":
        return (df['winner'] == 'Villagers').mean()
    if metric == "vote_precision":
        sub = df[df['total_villager_votes'] > 0]
        total_votes = sub['total_villager_votes'].sum()
        wolf_votes = sub['votes_against_wolf'].sum()
        return (wolf_votes / total_votes) if total_votes > 0 else 0.0
    raise ValueError(f"proportion is for proportion metrics only, got {metric!r}")


def cohens_d(coach_vals, base_vals):
    """Cohen's d with pooled SD; positive => coach mean higher than baseline."""
    n1, n2 = len(coach_vals), len(base_vals)
    if n1 < 2 or n2 < 2:
        return float('nan')
    v1, v2 = np.var(coach_vals, ddof=1), np.var(base_vals, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return (np.mean(coach_vals) - np.mean(base_vals)) / pooled_sd


def cohens_h(p_coach, p_base):
    """Cohen's h for two proportions (e.g. Win Rate, Vote Precision)."""
    phi = lambda p: 2.0 * np.arcsin(np.sqrt(np.clip(p, 0.0, 1.0)))
    return phi(p_coach) - phi(p_base)


def magnitude_label(x):
    a = abs(x)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


print("\n\n" + "#" * 60)
print("# EFFECT SIZES (coach vs baseline)")
print("#  Cohen's h for proportion metrics (Win Rate, Vote Precision);")
print("#  Cohen's d for continuous metrics (Belief Error).")
print("#" * 60)

for base_scn, coach_scn in DIFFERENCE_PAIRS:
    df_base = load_scenario_df(base_scn)
    df_coach = load_scenario_df(coach_scn)
    if df_base is None or df_coach is None:
        continue

    print(f"\n=== {coach_scn}  vs  {base_scn} ===")
    print("-" * 64)
    for key, label, _dp, is_prop, _divisor in DIFF_METRICS:
        if is_prop:
            # Proportion metric -> Cohen's h on the aggregate proportions.
            h = cohens_h(proportion(df_coach, key), proportion(df_base, key))
            print(f"{label:<16} Cohen's h = {h:+.3f}  ({magnitude_label(h)})")
        else:
            # Continuous metric -> Cohen's d on per-game values.
            d = cohens_d(per_game_values(df_coach, key), per_game_values(df_base, key))
            print(f"{label:<16} Cohen's d = {d:+.3f}  ({magnitude_label(d)})")
    print("  (positive = coach higher; for Belief Error, higher = worse)")
    print("=" * 64)