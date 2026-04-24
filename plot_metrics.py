import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

def calculate_single_game_metrics(game_dir: Path):
    metrics_path = game_dir / "metrics.json"
    summary_path = game_dir / "game_summary.json"
    directives_path = game_dir / "coach_directives.json"

    # Notice: No print statement here anymore! Just a silent skip.
    if not metrics_path.exists() or not summary_path.exists():
        return None

    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)
    
    with open(summary_path, "r") as f:
        summary_data = json.load(f)

    directives = {}
    if directives_path.exists():
        with open(directives_path, "r") as f:
            raw_directives = json.load(f)
            for role, d_list in raw_directives.items():
                for d in d_list:
                    directives[d["id"]] = d

    # 1. DCR: Directive Compliance Rate
    applicable_count = 0
    compliant_count = 0

    for player, events in metrics_data.items():
        for event in events:
            if event.get("had_applicable_directive"):
                applicable_count += 1
                directive_id = event.get("directive_id")
                action_summary = event.get("action_summary", "").lower()
                
                is_compliant = False
                if directive_id in directives:
                    expected_action = directives[directive_id]["action"].lower()
                    core_verbs = [word for word in expected_action.split("_") if word]
                    if any(verb in action_summary for verb in core_verbs):
                        is_compliant = True
                else:
                    if "abstained" not in action_summary and "nothing" not in action_summary:
                        is_compliant = True

                if is_compliant:
                    compliant_count += 1

    dcr = (compliant_count / applicable_count) if applicable_count > 0 else None

    # 2. IEI: Information Efficiency Index (Seer specific)
    roles = summary_data.get("roles", {})
    seer_name = next((name for name, role in roles.items() if role == "Seer"), None)
    
    iei_numerator = 0
    iei_denominator = 0

    if seer_name and seer_name in metrics_data:
        known_wolves = set()
        
        for round_str, round_data in summary_data.items():
            if not round_str.isdigit():
                continue
            round_num = int(round_str)
            
            for event in metrics_data[seer_name]:
                if event.get("phase") == "NIGHT" and event.get("investigated_target"):
                    target = event["investigated_target"]
                    if roles.get(target) == "Werewolf":
                        known_wolves.add(target)
            
            alive_players = get_alive_players_for_round(summary_data, round_num)
            active_known_wolves = {w for w in known_wolves if w in alive_players}

            if active_known_wolves:
                day_votes = round_data.get("votes", [])
                seer_vote = next((v["vote_for"] for v in day_votes if v["voter"] == seer_name), None)
                
                if seer_vote is not None:
                    iei_denominator += 1
                    if seer_vote in active_known_wolves:
                        iei_numerator += 1

    iei = (iei_numerator / iei_denominator) if iei_denominator > 0 else None

    # 3. DRR: Deception Resistance Rate
    wolves = {name for name, role in roles.items() if role == "Werewolf"}
    villagers = {name for name, role in roles.items() if role != "Werewolf"}
    
    bandwagon_votes = 0
    total_exposed_villager_votes = 0

    for round_str, round_data in summary_data.items():
        if not round_str.isdigit():
            continue
            
        day_votes = round_data.get("votes", [])
        wolf_targets = set()
        
        for vote in day_votes:
            if vote["voter"] in wolves and vote["vote_for"] in villagers:
                wolf_targets.add(vote["vote_for"])
                
        if wolf_targets:
            for vote in day_votes:
                if vote["voter"] in villagers:
                    total_exposed_villager_votes += 1
                    if vote["vote_for"] in wolf_targets:
                        bandwagon_votes += 1

    drr = 1.0 - (bandwagon_votes / total_exposed_villager_votes) if total_exposed_villager_votes > 0 else None

    return {"DCR": dcr, "IEI": iei, "DRR": drr}

def get_alive_players_for_round(summary_data, target_round):
    all_players = set(summary_data.get("roles", {}).keys())
    dead_players = set()
    for r in range(1, target_round):
        round_data = summary_data.get(str(r), {})
        dead_players.update(round_data.get("night_eliminated", []))
        if round_data.get("day_exiled"):
            dead_players.add(round_data["day_exiled"])
    return all_players - dead_players

def analyze_scenario(scenario_dir: Path):
    if not scenario_dir.exists():
        print(f"Error: Directory {scenario_dir} does not exist.")
        return

    game_dirs = sorted([d for d in scenario_dir.iterdir() if d.is_dir() and d.name.startswith("game_")])
    
    if not game_dirs:
        print(f"No game directories found in {scenario_dir}.")
        return

    game_ids = []
    dcr_history = []
    iei_history = []
    drr_history = []

    print(f"Analyzing {len(game_dirs)} games in {scenario_dir.name}...")

    for g_dir in game_dirs:
        metrics = calculate_single_game_metrics(g_dir)
        if metrics:
            game_num = int(g_dir.name.split("_")[-1])
            game_ids.append(game_num)
            
            dcr_history.append(metrics["DCR"] if metrics["DCR"] is not None else 0)
            iei_history.append(metrics["IEI"] if metrics["IEI"] is not None else 0)
            drr_history.append(metrics["DRR"] if metrics["DRR"] is not None else 0)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(game_ids, dcr_history, label="DCR (Directive Compliance Rate)", marker='o', linestyle='-', alpha=0.8, color='blue')
    ax.plot(game_ids, iei_history, label="IEI (Information Efficiency Index)", marker='s', linestyle='-', alpha=0.8, color='green')
    ax.plot(game_ids, drr_history, label="DRR (Deception Resistance Rate)", marker='^', linestyle='-', alpha=0.8, color='red')

    ax.set_title(f"Strategic Metric Evolution over Time: {scenario_dir.name}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Game Number", fontsize=12)
    ax.set_ylabel("Metric Rate (0.0 to 1.0)", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right', framealpha=0.9)

    plt.tight_layout()
    output_path = scenario_dir / "metrics_trend.png"
    plt.savefig(output_path, dpi=300)
    print(f"Done! Plot saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and visualize metrics across all games in a scenario.")
    parser.add_argument("scenario_dir", type=str, help="Path to the scenario directory")
    args = parser.parse_args()
    analyze_scenario(Path(args.scenario_dir))