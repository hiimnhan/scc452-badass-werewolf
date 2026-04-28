from config import SCENARIO_CONFIG
from constants import RESULTS_SUMMARY_FILENAME, FIGURES_FOLDERNAME, BIN_SIZE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# /opt/miniconda3/envs/scc452-badass-werewolf/bin/python -m evaluation.win_rate

all_data = []
GAME_BIN_AXIS_NAME = "Game"
WIN_RATE_AXIS_NAME = 'Villager Win Rate (%)'

# Process each scenario
for scenario in SCENARIO_CONFIG:
    file_path = (Path(__file__).parent.parent / "game_logs" / scenario / RESULTS_SUMMARY_FILENAME).resolve()
    try:
        df = pd.read_csv(file_path)
    except:
        continue
        
    # Create bins (e.g., Games 1-25 -> Bin 0, Games 26-50 -> Bin 1)
    df['Bin_Index'] = (df['game_id'].astype(int) - 1) // BIN_SIZE
    
    # Create readable labels for the x-axis (e.g., "1-25", "26-50")
    df[GAME_BIN_AXIS_NAME] = df['Bin_Index'].apply(lambda x: f"{x * BIN_SIZE + 1}-{(x + 1) * BIN_SIZE}")
    
    # Identify Villager wins 
    # NOTE: Change 'Winner' and 'Villagers' if your column/values are named differently
    df['Villager_Win'] = (df['winner'] == 'Villagers').astype(int)
    
    # Calculate win rate per bin
    win_rates = df.groupby(['Bin_Index', GAME_BIN_AXIS_NAME])['Villager_Win'].mean().reset_index()
    
    # Convert to percentage (0-100%)
    win_rates[WIN_RATE_AXIS_NAME] = win_rates['Villager_Win'] * 100 
    win_rates['Scenario'] = scenario
    
    all_data.append(win_rates)

# Combine into a single DataFrame
plot_df = pd.concat(all_data, ignore_index=True)

# Sort by actual index to ensure the x-axis is in chronological order
plot_df = plot_df.sort_values(by='Bin_Index')

# 4. Plot the Clustered Bar Chart
plt.figure(figsize=(12, 6))

# seaborn's barplot automatically creates a clustered chart when using 'hue'
ax = sns.barplot(
    data=plot_df, 
    x=GAME_BIN_AXIS_NAME, 
    y=WIN_RATE_AXIS_NAME, 
    hue='Scenario',
    hue_order=SCENARIO_CONFIG,
    palette='Set2' # Feel free to change the color palette
)

for container in ax.containers:
    # fmt='%.1f%%' formats the number to 1 decimal place with a % sign
    # padding=3 adds a tiny gap between the bar and the text
    ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=9)

# Customize the aesthetics
plt.title('Villager Win Rate per 25-Game Bin by Experiment', fontsize=16, fontweight='bold', pad=15)
plt.xlabel(GAME_BIN_AXIS_NAME, fontsize=12, fontweight='bold')
plt.ylabel(WIN_RATE_AXIS_NAME, fontsize=12, fontweight='bold')

# Move the legend outside the chart to avoid covering data
plt.legend(title='Scenario', loc='upper right')

# Set y-axis to always show 0 to 100%
plt.ylim(0, 100)  
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout to fit the legend
plt.tight_layout()

output_path = (Path(__file__).parent.parent / FIGURES_FOLDERNAME / "win_rate.png").resolve()

# Save the figure BEFORE plt.show()
# dpi=300 makes it high-resolution (good for presentations/papers)
# bbox_inches='tight' ensures the outside legend isn't cut off
plt.savefig(output_path, dpi=300, bbox_inches='tight')

print(f"Success! Chart saved to: {output_path}")

# Display the chart
# plt.show()