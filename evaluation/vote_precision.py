from config import SCENARIO_CONFIG
from constants import FIGURES_FOLDERNAME, GAME_SUMMARY_FILENAME, BIN_SIZE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import json
from run import VILLAGER_PLAYERS, WOLF_PLAYERS

# /opt/miniconda3/envs/scc452-badass-werewolf/bin/python -m evaluation.vote_precision

all_data = []

# ------------------------------------------------------------
# 1. Data Extraction
# ------------------------------------------------------------
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
        
        if not summary_path.exists():
            continue
            
        with open(summary_path, 'r') as file:
            game_summary = json.load(file)
        
        total_villager_votes = 0
        total_villager_votes_against_wolf = 0
        
        for round_key in game_summary:
            if "votes" in game_summary[round_key]:
                for vote in game_summary[round_key]["votes"]:
                    voter = vote['voter']
                    target = vote['vote_for']
                    
                    if voter in VILLAGER_PLAYERS:
                        total_villager_votes += 1
                        
                        if target in WOLF_PLAYERS:
                            total_villager_votes_against_wolf += 1
                            
        # Store the RAW counts instead of the calculated precision
        all_data.append({
            "Scenario": scenario,
            "Game_ID": game_id,
            "Total_Votes": total_villager_votes,
            "Votes_Against_Wolf": total_villager_votes_against_wolf
        })

# Convert to Pandas DataFrame
df = pd.DataFrame(all_data)

# ------------------------------------------------------------
# 2. Data Processing & Binning
# ------------------------------------------------------------
# Create the 25-game Bins (Games 1-25 -> Bin 0, Games 26-50 -> Bin 1, etc.)
df['Bin_Index'] = (df['Game_ID'] - 1) // BIN_SIZE
df['Bin_Label'] = df['Bin_Index'].apply(lambda x: f"{x * BIN_SIZE + 1}-{(x + 1) * BIN_SIZE}")

# Group by Scenario and Bin, then SUM the raw votes
binned_df = df.groupby(['Scenario', 'Bin_Index', 'Bin_Label'])[['Total_Votes', 'Votes_Against_Wolf']].sum().reset_index()

# Calculate the true Vote Precision for the bin (with division-by-zero protection)
binned_df['Vote_Precision (%)'] = binned_df.apply(
    lambda row: (row['Votes_Against_Wolf'] / row['Total_Votes'] * 100) if row['Total_Votes'] > 0 else 0, 
    axis=1
)

# Sort to ensure bins appear in order from left to right
binned_df = binned_df.sort_values(by='Bin_Index')

# ------------------------------------------------------------
# 3. Plotting Setup: Lock the color palette to original scenarios
# ------------------------------------------------------------
# Handle SCENARIO_CONFIG whether it's a list or dictionary
if isinstance(SCENARIO_CONFIG, dict):
    all_scenarios = list(SCENARIO_CONFIG.keys())
else:
    all_scenarios = list(SCENARIO_CONFIG)

# Map the 'Set2' colors to all original scenarios to keep colors consistent
all_colors = sns.color_palette('Set2', len(all_scenarios))
locked_palette = {scenario: color for scenario, color in zip(all_scenarios, all_colors)}

# ------------------------------------------------------------
# 4. Reusable Plotting Function
# ------------------------------------------------------------
def plot_precision_subset(data, target_scenarios, title, filename):
    plt.figure(figsize=(12, 6))

    ax = sns.barplot(
        data=data, 
        x='Bin_Label', 
        y='Vote_Precision (%)', 
        hue='Scenario',
        hue_order=target_scenarios,  # Only plot the requested scenarios
        palette=locked_palette       # Use the locked colors
    )

    # Add precision labels on top of the bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=9)

    # Customize Aesthetics
    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Game Bin', fontsize=12, fontweight='bold')
    plt.ylabel('Vote Precision (%)', fontsize=12, fontweight='bold')

    # Legend inside the plot at the top right
    plt.legend(title='Scenario', loc='upper right')

    # Set y-axis slightly above 100% so text doesn't get cut off
    plt.ylim(0, 110)  
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Ensure output directory exists
    output_dir = (Path(__file__).parent.parent / FIGURES_FOLDERNAME).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / filename

    # Save and close the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Success! Chart saved to: {output_file}")
    plt.close()

# ------------------------------------------------------------
# 5. Generate the Charts
# ------------------------------------------------------------

# Chart 1: 12B Scenarios
plot_precision_subset(
    data=binned_df,
    target_scenarios=["baseline_12b_31B", "coach_12b_31B"],
    # title='Villager Vote Precision per 25-Game Bin (12B Models)',
    title='',
    filename="vote_precision_12b.png"
)

# Chart 2: 31B Scenarios
plot_precision_subset(
    data=binned_df,
    target_scenarios=["baseline_31B_31B", "coach_31B_31B"],
    # title='Villager Vote Precision per 25-Game Bin (31B Models)',
    title='',
    filename="vote_precision_31b.png"
)

# Chart 3: All Scenarios
plot_precision_subset(
    data=binned_df,
    target_scenarios=all_scenarios,
    # title='Villager Vote Precision per 25-Game Bin (All Scenarios)',
    title='',
    filename="vote_precision_all.png"
)