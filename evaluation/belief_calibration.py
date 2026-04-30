from config import SCENARIO_CONFIG
from constants import FIGURES_FOLDERNAME, GAME_SUMMARY_FILENAME, SUSPICION_LOG_FILENAME, BIN_SIZE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import json
from run import PLAYERS, VILLAGER_PLAYERS, WOLF_PLAYERS
import math

# /opt/miniconda3/envs/scc452-badass-werewolf/bin/python -m evaluation.belief_calibration

all_data = []

# Process each scenario
for scenario in SCENARIO_CONFIG:
    scenario_game_dir = (Path(__file__).parent.parent / "game_logs" / scenario).resolve()
    
    if not scenario_game_dir.exists():
        continue
    
    for game_dir in scenario_game_dir.iterdir():
        if not game_dir.is_dir():
            continue
            
        game_id_str = game_dir.name[-3:]
        if not game_id_str.isdigit():
            continue
            
        game_id = int(game_id_str)
        summary_path = (game_dir / GAME_SUMMARY_FILENAME.replace(".md", ".json")).resolve()
        suspicion_path = (game_dir / SUSPICION_LOG_FILENAME).resolve()
        
        if not summary_path.exists() or not suspicion_path.exists():
            continue
            
        with open(summary_path, 'r') as file:
            game_summary = json.load(file)
        
        with open(suspicion_path, 'r') as file:
            suspicion_log = json.load(file)
        
        # ------------------------------------------------------------
        # 1. Initialize Game Tracking
        total_game_error = 0
        total_game_comparisons = 0
        
        # CRITICAL: Create a fresh copy of the players list for this specific game
        alive_players = list(PLAYERS) 
        
        for round_key, player_suspicion in suspicion_log.items():
            # ALLOW "end_game" to pass through the filter
            if not round_key.isdigit() and round_key != "end_game":
                continue
                
            round_data = game_summary.get(round_key, {})
            
            # --- Record Suspicion (SKIPPED FOR ROUND 1) ---
            if round_key != "1":
                for player, suspicion in player_suspicion.items():
                    if (player not in VILLAGER_PLAYERS) or (player not in alive_players):
                        continue
                    
                    for target, score_data in suspicion.items():
                        if (target == player) or (target not in alive_players):
                            continue
                        
                        is_wolf = 1 if target in WOLF_PLAYERS else 0
                        predicted_score = score_data["score"]
                        
                        total_game_error += (predicted_score - is_wolf) ** 2
                        total_game_comparisons += 1
            
            # --- Update elimination for next round ---
            night_elim = round_data.get("night_eliminated")
            if night_elim and night_elim in alive_players:
                alive_players.remove(night_elim)
                
            day_exiled = round_data.get("day_exiled")
            if day_exiled and day_exiled in alive_players:
                alive_players.remove(day_exiled)

        # Store the RAW totals for this game
        all_data.append({
            "Scenario": scenario,
            "Game_ID": game_id,
            "Total_Error": total_game_error,
            "Total_Comparisons": total_game_comparisons
        })
        
# ------------------------------------------------------------
# 2. Data Processing & Binning
df = pd.DataFrame(all_data)

# Create the 25-game Bins
df['Bin_Index'] = (df['Game_ID'] - 1) // BIN_SIZE
df['Bin_Label'] = df['Bin_Index'].apply(lambda x: f"{x * BIN_SIZE + 1}-{(x + 1) * BIN_SIZE}")

# Group by Scenario and Bin, summing the raw values
binned_df = df.groupby(['Scenario', 'Bin_Index', 'Bin_Label'])[['Total_Error', 'Total_Comparisons']].sum().reset_index()

# Calculate Mean Squared Error (MSE) for the bin
binned_df['RMSE'] = binned_df.apply(
    lambda row: math.sqrt(row['Total_Error'] / row['Total_Comparisons']) if row['Total_Comparisons'] > 0 else 0, 
    axis=1
)

binned_df = binned_df.sort_values(by='Bin_Index')

# ------------------------------------------------------------
# 3. Plotting
plt.figure(figsize=(12, 6))

ax = sns.barplot(
    data=binned_df, 
    x='Bin_Label', 
    y='RMSE', 
    hue='Scenario',
    hue_order=SCENARIO_CONFIG,
    palette='Set2'
)

# Add error labels on top of the bars
for container in ax.containers:
    # fmt='%.3f' shows 3 decimal places (e.g., 0.145) since errors are usually small fractions
    ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)

# plt.title('Villager Belief Error (RMSE) per 25-Game Bin', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Game Bin', fontsize=12, fontweight='bold')
plt.ylabel('Belief RMSE', fontsize=12, fontweight='bold')

plt.legend(title='Scenario', loc='upper right')

# Dynamically set y-axis limit to 15% above the maximum MSE so labels fit
# max_mse = binned_df['RMSE'].max()
# plt.ylim(0, max_mse * 1.15)
plt.ylim(0, 1)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the Figure
output_dir = (Path(__file__).parent.parent / FIGURES_FOLDERNAME).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "belief_calibration.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')

print(f"Success! Chart saved to: {output_file}")

# plt.show()