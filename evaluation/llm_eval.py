import json
import re
import time
import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage
from utils import get_llm
from pathlib import Path
from run import VILLAGER_PLAYERS
from constants import PLAYER_FINAL_STRATEGY_FILENAME, COACH_FEEDBACK_FILENAME, GAME_SUMMARY_FILENAME
from config import SCENARIO_CONFIG
from dotenv import load_dotenv
import os
import openai

load_dotenv()

# /opt/miniconda3/envs/scc452-badass-werewolf/bin/python -m evaluation.llm_eval

class GameEvaluator:
    def __init__(self):
        self._model = get_llm("o3-mini", api_key=os.environ.get("OPENAI_API_KEY"))

    def _call_model(self, system: str, prompt: str, max_tokens: int = 10000, timeout: int = 200) -> dict:
        """Your existing model calling function."""
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ]
        kwargs = {"timeout": timeout}

        if "GoogleGenerativeAI" not in type(self._model).__name__:
            kwargs["max_tokens"] = max_tokens

        resp = self._model.invoke(messages, **kwargs).content
        if isinstance(resp, list):
            resp = " ".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in resp).strip()
        elif isinstance(resp, str):
            resp = resp.strip()

        result: dict = {}
        clean_resp = resp.strip()
        match = re.search(r"(\{.*\})", clean_resp, re.DOTALL)
        if match:
            clean_resp = match.group(1)

        try:
            result = json.loads(clean_resp)
        except json.JSONDecodeError:
            result = {"raw": resp}

        result.setdefault("_prompt", prompt)
        return result

    def evaluate_game_metrics(self, game_number: int, player_name: str, 
                              prev_strategy: str, prev_coach_feedback: str, 
                              current_strategy: str, current_game_summary: str) -> dict:
        """
        Evaluates Compliance Rate and Summarization Ability in one LLM pass.
        """
        system_prompt = (
            "You are an expert behavioral analyst evaluating an AI agent's gameplay in the Werewolf game. "
            "Your task is to analyze the provided game logs and output a strict JSON evaluating the player's "
            "Compliance Rate and Summarization Ability. Do not output any markdown formatting outside of the JSON block. "
            "Return ONLY valid JSON."
        )

        user_prompt = f"""
Analyze the following gameplay data for {player_name} in Game {game_number}.

--- INPUT DATA ---
Game Number: {game_number}
Player: {player_name}
Previous Game Strategy (Game {game_number - 1}): {prev_strategy if prev_strategy else "N/A"}
Coach Feedback (After Game {game_number - 1}): {prev_coach_feedback}
Current Game Strategy (Game {game_number}): {current_strategy}
Current Game Summary (Game {game_number}): {current_game_summary}

--- EVALUATION TASKS ---
1. Compliance Rate: Look at the 'Coach Feedback'. Break it down into specific directives. For each directive, check the 'Current Game Strategy' and 'Current Game Summary' to see if {player_name} actually executed it. Mark as "FOLLOW" or "IGNORE".
2. Summarization Ability: Compare the 'Current Game Strategy' against the 'Previous Game Strategy' + 'Coach Feedback'. 
   - Score (0-10): How well did the player synthesize the old strategy and the new feedback into their current strategy?
   - Overlap Score (0-1): How much does the updated information in the current strategy match the specific points raised in the coach's feedback?

--- EXPECTED JSON OUTPUT FORMAT ---
{{
  "game_number": {game_number},
  "player": "{player_name}",
  "compliance": [
    {{
      "directive": "Brief summary of the coach's directive",
      "status": "FOLLOW or IGNORE",
      "reasoning": "1 sentence explaining why"
    }}
  ],
  "summarization_ability": {{
    "summarization_score": <int 0-10>,
    "feedback_overlap_score": <float 0-1>,
    "reasoning": "Brief explanation of the scores"
  }}
}}
"""
        return self._call_model(system=system_prompt, prompt=user_prompt, max_tokens=3000)


def evaluate_with_retry(evaluator: GameEvaluator, *args, max_retries=50):
    """Safeguard: Retries the LLM call until valid JSON with expected keys is returned."""
    for attempt in range(max_retries):
        try:
            result = evaluator.evaluate_game_metrics(*args)
            
            # Check if the parsing failed or if crucial keys are missing
            if "raw" not in result and "compliance" in result and "summarization_ability" in result:
                return result
                
            print(f"      [!] Attempt {attempt + 1}/{max_retries} returned invalid JSON. Retrying...")
            
        except openai.BadRequestError as e:
            # Catches the max_tokens limit error (and other 400 errors)
            print(f"      [!] API Token Error on Attempt {attempt + 1}/{max_retries}: {e.message}")
            
        except Exception as e:
            # Catches unexpected network disconnects or 500 server errors
            print(f"      [!] Unexpected API Error on Attempt {attempt + 1}/{max_retries}: {e}")

        # Brief pause before retrying
        time.sleep(2) 
        
    raise ValueError("LLM failed to return correctly formatted JSON after maximum retries.")


# --- MAIN SCRIPT EXECUTION ---

game_evaluator = GameEvaluator()

# Setup CSV tracking
csv_path = Path(__file__).parent / "llm_eval.csv"
completed_games = set()

# Load existing data to skip previously completed games
if csv_path.exists():
    existing_df = pd.read_csv(csv_path)
    # If a game is in the CSV, it means all villagers were successfully evaluated
    for _, row in existing_df.iterrows():
        completed_games.add((row['scenario'], row['game_id']))


for scenario in SCENARIO_CONFIG:
    if not scenario.startswith("coach"):
        continue
    
    scenario_dir = (Path(__file__).parent.parent / "game_logs" / scenario).resolve()
    
    if not scenario_dir.exists():
        continue
    
    for game_dir in sorted(scenario_dir.iterdir()):
        if not game_dir.is_dir():
            continue
            
        game_id_str = game_dir.name[-3:]
        if not game_id_str.isdigit():
            continue
        
        game_id = int(game_id_str)
        if game_id < 2:
            continue
            
        # 1. Skip if already evaluated
        if (scenario, game_id) in completed_games:
            print(f"Skipping {scenario} - Game {game_id} (Already completed)")
            continue
            
        print(f"Evaluating {scenario} - Game {game_id}...")
        
        # 2. Retrieve game-level files
        try:
            with open(game_dir / GAME_SUMMARY_FILENAME) as file:
                current_game_summary = file.read()
            
            with open(scenario_dir / f"game_{game_id-1:03d}" / COACH_FEEDBACK_FILENAME) as file:
                previous_game_coach_feedback = file.read()
        except FileNotFoundError as e:
            print(f"  [!] Missing game-level file: {e}. Skipping game.")
            continue
        
        # 3. Evaluate each player
        game_results_batch = []
        evaluation_failed = False
        
        for player in VILLAGER_PLAYERS:
            player_strategy_file_name = PLAYER_FINAL_STRATEGY_FILENAME.format(name=player)
            
            try:
                with open(game_dir / "player_strategies" / player_strategy_file_name) as file:
                    current_game_strategy = file.read()
                
                with open(scenario_dir / f"game_{game_id-1:03d}" / "player_strategies" / player_strategy_file_name) as file:
                    previous_game_strategy = file.read()
                    
                # LLM Call with Retry Safeguard
                print(f"    -> Running LLM for {player}...")
                metrics = evaluate_with_retry(
                    game_evaluator,
                    game_id, 
                    player, 
                    previous_game_strategy, 
                    previous_game_coach_feedback, 
                    current_game_strategy, 
                    current_game_summary
                )
                
                # Flatten the data for the CSV
                game_results_batch.append({
                    "scenario": scenario,
                    "game_id": game_id,
                    "player": player,
                    "compliance_data": json.dumps(metrics.get("compliance", [])),
                    "summarization_score": metrics.get("summarization_ability", {}).get("summarization_score"),
                    "feedback_overlap_score": metrics.get("summarization_ability", {}).get("feedback_overlap_score"),
                    "summarization_reasoning": metrics.get("summarization_ability", {}).get("reasoning")
                })
                
            except FileNotFoundError as e:
                print(f"  [!] Missing player file: {e}. Skipping entire game to maintain atomic write.")
                evaluation_failed = True
                break
            except ValueError as e:
                print(f"  [!] {e} Skipping entire game.")
                evaluation_failed = True
                break
                
        # 4. Atomic Write: Only save if all players in the game succeeded
        if not evaluation_failed and len(game_results_batch) == len(VILLAGER_PLAYERS):
            df_new = pd.DataFrame(game_results_batch)
            
            # Write header only if the file doesn't exist yet
            write_header = not csv_path.exists()
            df_new.to_csv(csv_path, mode='a', index=False, header=write_header)
            
            completed_games.add((scenario, game_id))
            print(f"  [+] Successfully saved {scenario} - Game {game_id} to CSV.")