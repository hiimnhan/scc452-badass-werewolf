import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from constants import BIN_SIZE
from run import SCENARIO_CONFIG

# /opt/miniconda3/envs/scc452-badass-werewolf/bin/python -m evaluation.llm_metrics

# --- 1. Load and Parse Data ---
llm_eval = pd.read_csv("evaluation/llm_eval.csv")

def get_compliance_counts(json_string):
    total_directives = 0
    follow_count = 0
    ignore_count = 0
    
    if pd.isna(json_string):
        return follow_count, ignore_count, total_directives
        
    try:
        data = json.loads(json_string)
        for directive in data:
            total_directives += 1
            status = directive.get('status') 
            
            if status == "FOLLOW":
                follow_count += 1
            elif status == "IGNORE":
                ignore_count += 1
    except (json.JSONDecodeError, TypeError):
        pass 
        
    return follow_count, ignore_count, total_directives

llm_eval[['follow_count', 'ignore_count', 'total_directives']] = llm_eval['compliance_data'].apply(
    lambda x: pd.Series(get_compliance_counts(x))
)

# --- 2. Create Bins & Aggregate ---

# Create the 25-game Bins based on game_id
llm_eval['Bin_Index'] = (llm_eval['game_id'] - 1) // BIN_SIZE
llm_eval['Bin_Label'] = llm_eval['Bin_Index'].apply(lambda x: f"{x * BIN_SIZE + 1}-{(x + 1) * BIN_SIZE}")

# Group by Scenario and Bin, aggregating the sums and row counts
binned_df = llm_eval.groupby(['scenario', 'Bin_Index', 'Bin_Label']).agg(
    total_follow=('follow_count', 'sum'),
    total_directives=('total_directives', 'sum'),
    total_summarization=('summarization_score', 'sum'),
    total_overlap=('feedback_overlap_score', 'sum'),
    row_count=('game_id', 'count') # Counts the number of rows in this bin
).reset_index()

# --- 3. Calculate Final Metrics ---
# Compliance Rate (%) = total_follow / total_directives * 100
binned_df['Compliance_Rate'] = binned_df.apply(
    lambda row: (row['total_follow'] / row['total_directives'] * 100) if row['total_directives'] > 0 else 0, 
    axis=1
)

# Summarization Ability = total summarization_score / number of rows
binned_df['Summarization_Ability'] = binned_df['total_summarization'] / binned_df['row_count']

# Feedback Overlap Proportion = total feedback_overlap_score / number of rows
binned_df['Feedback_Overlap (%)'] = (binned_df['total_overlap'] / binned_df['row_count']) * 100

# Sort to ensure chronological order
binned_df = binned_df.sort_values(by='Bin_Index')

# --- 4. Plotting Function ---
# Ensure output directory exists
output_dir = Path("figures").resolve()
output_dir.mkdir(parents=True, exist_ok=True)

def plot_metric(data, y_col, title, ylabel, filename, is_percentage=False, y_max=None):
    plt.figure(figsize=(12, 6))
    
    scenario_keys = list(SCENARIO_CONFIG.keys())
    target_scenarios = [scenario_keys[1], scenario_keys[3]]
    all_colors = sns.color_palette('Set2', len(scenario_keys))
    locked_palette = {scenario: color for scenario, color in zip(scenario_keys, all_colors)}
    # ----------------------

    ax = sns.barplot(
        data=data, 
        x='Bin_Label', 
        y=y_col, 
        hue='scenario',
        hue_order=target_scenarios,  # Forces it to only show the 2nd and 4th
        palette=locked_palette       # Forces them to use their original colors
    )
    
    # Add labels on top of bars
    for container in ax.containers:
        fmt_str = '%.1f%%' if is_percentage else '%.2f'
        ax.bar_label(container, fmt=fmt_str, padding=3, fontsize=9)
        
    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Game Bin', fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    
    plt.legend(title='Scenario', loc='upper right')
    
    # Set y-axis limits
    if y_max:
        plt.ylim(0, y_max)
    else:
        # Dynamically set y-axis limit to 15% above the max value to fit labels
        plt.ylim(0, data[y_col].max() * 1.15)
        
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close() # Good practice to close the figure so they don't stack in memory

# --- 5. Generate Visuals ---
# Chart 1: Compliance Rate
plot_metric(
    data=binned_df, 
    y_col='Compliance_Rate', 
    title='Villager Compliance Rate per 25-Game Bin', 
    ylabel='Compliance Rate (%)', 
    filename='compliance_rate.png',
    is_percentage=True,
    y_max=110 # Max percentage + headroom
)

# Chart 2: Summarization Ability
plot_metric(
    data=binned_df, 
    y_col='Summarization_Ability', 
    title='Villager Summarization Ability per 25-Game Bin', 
    ylabel='Summarization Score (0-10)', 
    filename='summarization_ability.png',
    is_percentage=False,
    y_max=11 # Max score of 10 + headroom
)

# Chart 3: Feedback Overlap Proportion
plot_metric(
    data=binned_df, 
    y_col='Feedback_Overlap (%)',               # Updated column name
    title='Feedback Overlap Proportion per 25-Game Bin', 
    ylabel='Overlap Proportion (%)',            # Updated y-axis label
    filename='feedback_overlap.png',
    is_percentage=True,                         # Automatically formats to %.1f%%
    y_max=115                                   # Updated headroom for a 0-100 scale
)