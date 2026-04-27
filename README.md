# Installation

## 1. Clone and setup environment

```bash
git clone <repository-url>
cd scc452-badass-werewolf
conda env create -f env.yaml
conda activate scc452-badass-werewolf
```

## 2. Set up environment variables

Create a `.env` file and set the required API keys:

```bash
cp .env.example .env   # if .env.example exists, otherwise create manually
```

Required keys

```
DEEPINFRA_API_KEY=...
OPENROUTER_API_KEY=...
GEMINI_API_KEY=...
```

## 3. Run experiments

```bash
python run.py --scenario <scenario> --mode <override|append> --games <n>
```

### Arguments

| Flag         | Short | Default            | Description                                                                                           |
| ------------ | ----- | ------------------ | ----------------------------------------------------------------------------------------------------- |
| `--scenario` | `-s`  | `baseline_12b_31B` | Experiment scenario (see below)                                                                       |
| `--games`    | `-g`  | `100`              | Number of games to run                                                                                |
| `--mode`     | `-m`  | `append`           | `override`: wipe existing results and start from game_001. `append`: resume from last completed game. |

### Available scenarios

| Scenario           | Villager model | Wolf model  | Coaching |
| ------------------ | -------------- | ----------- | -------- |
| `baseline_12b_31B` | gemma-3-12b    | gemma-4-31B | No       |
| `coach_12b_31B`    | gemma-3-12b    | gemma-4-31B | Yes      |
| `baseline_31B_31B` | gemma-4-31B    | gemma-4-31B | No       |
| `coach_31B_31B`    | gemma-4-31B    | gemma-4-31B | Yes      |

### Examples

```bash
# Start fresh, run 20 games of the 31B baseline
python run.py --scenario baseline_31B_31B --mode override --games 20

# Resume an interrupted coaching run
python run.py --scenario coach_31B_31B --mode append --games 100
```

### Auto-restart on crash

`watch_run.py` wraps `run.py` and restarts it automatically if it crashes, resuming from the last completed game:

```bash
python watch_run.py --scenario coach_31B_31B --mode append --games 100
```

## 4. Output

```
game_logs/
  {scenario}/
    results_summary.csv          win/loss record across all games
    game_001/
      game_summary.txt           public announcements in order
      debate_log.txt             every statement from every day debate
      vote_log.txt               every vote cast across all rounds
      roles_this_game.txt        who played which role
      {name}_{role}_note.txt     each player's compiled notes
      coach_feedback.txt         coach post-game feedback (if coaching enabled)

strategies/
  {scenario}/
    {name}_strategy.txt          each player's accumulated strategy
    coach_strategy.txt           coach's accumulated strategy
```
