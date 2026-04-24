# Analysis of the Coach–Player Paradigm in an LLM Werewolf Simulation

## 1. Abstract

This report analyses the experimental scenario `coach_and_self_analyze`, in which villager-side agents receive post-game feedback from an external Coach and additionally perform self-analysis, while werewolf-side agents perform self-analysis only. The central hypothesis is that the coach–player paradigm should close the performance gap between the two sides. Empirically the wolves continue to dominate. This report documents the system architecture, characterises the mechanisms by which coaching propagates (and fails to propagate) into agent behaviour, and enumerates the structural, informational, and LLM-capability factors that bound the effectiveness of coaching. The conclusion is that coaching is **necessary but not sufficient**: it produces high-quality strategy text but cannot overcome a model-scale asymmetry, a symmetric self-improvement loop on the adversary side, and rule-layer game asymmetries that favour the wolves.

## 2. System Architecture

The simulation is implemented in `game.py`, with per-role agents under `players/`. The architecture is a frozen-LLM multi-agent system with external text-based memory; no weights are updated during training.

### 2.1 Roles and win conditions

Roles are defined in `players/base_player.py:53–60`.

- Villager: no night action.
- Werewolf: knows fellow wolves; eliminates one non-wolf per night.
- Seer: learns one player's alignment per night.
- Guard: protects one player per night; cannot protect the same player twice consecutively.
- Witch: has one Save potion and one Poison potion.

Win conditions are evaluated in `game.py:101–120`:
- Villagers win when `len(werewolves) == 0`.
- Wolves win when `len(werewolves) >= len(villagers)` (parity is sufficient).

### 2.2 Phase flow

The `Phase` enum in `game.py:44–51` defines the state machine: `PROTECT → UNMASK → SAVE_OR_POISON → RESOLVE_NIGHT → CHECK_WINNER_NIGHT → DEBATE → VOTE → EXILE`.

At each night, wolves run a multi-round debate loop in `game.py:141–172`: up to three rounds, terminating when two consecutive wolf votes match. Agreement yields `gs._eliminated = target`. A fallback tiebreaker at `game.py:174–178` resolves disagreement by random choice.

### 2.3 Information channels

- `_wolf_debate_log` (`game.py:87`): private inter-wolf dialogue history. Visible only to wolves.
- `_wolf_target_log` (`game.py:88`): per-night targets.
- `_seer_log`, `_guard_log`, `_witch_log` (`game.py:89–91`): private per-role logs. Not shared between roles.
- `_game_summary_log` (`game.py:93`): public announcements only. Sole input to the Coach.

### 2.4 Wolf elimination prompt

In `players/wolf.py:28–62`, the wolf agent receives `wolf_teammates` explicitly and computes `targets = [p for p in alive_players if p != self._name and p not in wolf_teammates]`. Teammates are formatted into the elimination prompt together with the running dialogue history. Wolves therefore have a guaranteed-innocent list of size two from round one.

### 2.5 Suspicion initialisation

`players/base_player.py:209–226` seeds per-agent suspicion scores. Villagers start all other players at 0.5 (uniform prior). Wolves start fellow wolves at 0.0 and everyone else at 0.5. This encodes the information asymmetry at t = 0.

### 2.6 Strategy files as non-parametric memory

After each game, every agent updates a persistent `<name>_strategy.txt` file. The update logic is in `players/base_player.py:697–844`.

- Villager side (`_update_strategy_villager`, lines 737–796) may draw from three sources: own game record (self-analysis), coach feedback, and the current strategy file. Which sources are used depends on `self._self_analyze` and `self._coaching`.
- Wolf side (`_update_strategy_wolf`, lines 798–844) always self-analyses. Flags `self_analyze` and `coaching` are explicitly ignored for wolves (documented at lines 713–714).

`_write_strategy` at line 846 persists the new strategy and refreshes `self._setup_prompt` so that the next game conditions on the updated rules.

## 3. Experimental Configuration

### 3.1 Model assignment

From `config.py:31–32`:

```python
VILLAGER_MODEL = "google/gemma-3-4b-it"   # 4B parameters
WOLF_MODEL     = "google/gemma-4-31B-it"  # 31B parameters
```

The Coach is instantiated with `wolf_llm` in `run.py:244`, i.e. the 31B model. Coaching text quality is therefore not the bottleneck; executor capability is.

### 3.2 Scenarios

From `config.py:38–63`:

| Scenario               | coaching | self_analyze | deviation | personality |
|------------------------|----------|--------------|-----------|-------------|
| `baseline`             | False    | True         | False     | False       |
| `coach_and_self_analyze` | True  | True         | False     | False       |
| `deviate`              | True     | True         | True      | False       |
| `personality`          | True     | True         | False     | True        |

### 3.3 Experimental protocol

One hundred games were executed per scenario. Game outcomes, logs, and per-player strategy files are persisted under `game_logs/<scenario>/game_NNN/` and `strategies/<scenario>/` respectively. This report focuses on `coach_and_self_analyze`.

## 4. Observed Result

Across the 100 `coach_and_self_analyze` games, villager win rate remains low and wolf win rate remains dominant. Villager wins occur in a small minority of games, typically when the Witch's poison coincides with a wolf identified by the Seer, or when a procedural anomaly disrupts wolf execution. Wolf wins occur through consistent early elimination of power roles followed by narrative control of the day debate.

## 5. Why the Coach–Player Paradigm Under-Performs

The result is not that coaching fails to do anything — villager strategy files visibly improve across games, and the Coach's diagnoses in `coach_feedback.txt` are accurate — but that the lift is bounded by factors coaching cannot address. These factors divide into three layers: rule layer, capability layer, and paradigm layer.

### 5.1 Rule-layer asymmetries

These are properties of `game.py` and the role mechanics. No strategy text can rewrite them.

**5.1.1 Win-threshold imbalance.** Wolves win at parity; villagers must achieve zero wolves. In a 7-player game (5 villagers + 2 wolves), wolves need three successful kill rounds, whereas villagers must correctly identify and exile two wolves without net error. Because villager day-vote error is non-zero, and because wolves kill deterministically each night absent Guard/Witch intervention, the expected time-to-win is shorter for wolves.

**5.1.2 Power-role self-exposure.** The Seer's information has zero value unless communicated publicly. The moment it is communicated, wolves have a high-priority target. The Witch's potion use reveals Witch existence; the Guard's protection is inferable from death patterns (a targeted player surviving implies Guard intervention, which constrains Guard identity). The coach can prescribe *when* to reveal; it cannot prescribe *whether*, because revealing is operationally required.

**5.1.3 Guard has no Seer identifier.** The Coach strategy observed in game 80 prescribes "Guard must protect the Seer every single night." The Guard agent has no in-game channel to learn the Seer's identity. Guard success reduces to guessing over `1/|alive villagers|` at best. The coach's rule is not executable within the game rules.

**5.1.4 Wolf coordination vs villager role silos.** Wolves run an up-to-three-round debate with shared `dialogue_history` (`game.py:141–172`) before each kill, converging on a single target. The Seer, Guard, and Witch act independently each night with no shared log. The game provides no villager-side equivalent of `_wolf_debate_log`. Coaching cannot introduce a communication channel the engine does not expose.

**5.1.5 Asymmetric starting information.** Per 2.5 above, wolves start round one with two guaranteed-innocent allies and a narrowed target pool of `N−2`. Villagers start with a uniform `0.5` prior over `N−1` suspects. This is a rule-layer allocation that coaching cannot equalise.

### 5.2 Capability-layer asymmetries (LLM scale)

**5.2.1 Executor capability gap.** The villager-side executor is 4B; the wolf-side executor is 31B. The Coach (31B) produces high-quality strategy text, but villager agents must follow it at 4B capability. The pipeline is therefore bottlenecked not at strategy production but at strategy execution.

**5.2.2 Instruction-following fidelity.** At 4B, instruction following degrades under adversarial in-context pressure. Coach rules such as "reject vibe arguments" function as lexical filters; a 31B wolf trivially paraphrases banned vocabulary (e.g. "calculated deflection rooted in behavioural mirroring"), which the 4B villager fails to recognise as a semantic equivalent of the banned pattern. This is a standard adversarial prompt-robustness failure.

**5.2.3 In-context learning asymmetry.** The 31B wolf has a steeper ICL curve — it generalises new patterns from few in-game examples. The 4B villager adheres more literally to written rules and generalises less. Identical rule text in the two prompts does not produce identical policy.

**5.2.4 Trajectory-summarisation quality.** Self-analysis is a summarisation-over-trajectory task in which the agent extracts regularities from a full game record. Extraction quality scales with model capability and context handling. The 31B wolf produces higher-fidelity strategy updates per game than the 4B villager can absorb from the 31B Coach.

### 5.3 Paradigm-layer limits (the coach–player loop itself)

**5.3.1 Symmetric self-improvement on the adversary side.** `_update_strategy_wolf` (`players/base_player.py:798`) runs unconditionally. The wolves are not a static target; they self-analyse each game using the 31B model, writing to `strategies/coach_and_self_analyze/<wolf_name>_strategy.txt`. Per-game improvement is thus two-sided. The net closing rate of the gap depends on `Δ_villager` versus `Δ_wolf`; empirically `Δ_wolf > Δ_villager` because (a) the wolf analyser is 31B, (b) wolf analysis has access to both public `_game_summary_log` and private `_wolf_debate_log`, whereas the Coach sees only the public log, and (c) the deception space is larger than the cooperation space and supports more novel tactics per game.

**5.3.2 Coach input blindness.** The Coach analyses only `_game_summary_log`. The Coach does not see `_wolf_debate_log`, `_seer_log`, `_guard_log`, or `_witch_log`. The Coach inherits the same partial observability as the villagers. Coaching cannot transfer information the Coach itself does not possess.

**5.3.3 Post-hoc timing.** Coaching is applied between games, via strategy-file regeneration. Wolf manipulation occurs in-game, in-context. The static strategy file cannot respond dynamically to adversarial narrative shifts within a single game.

**5.3.4 Behavioural transparency of coach effects.** Wolves do not read coach files or villager strategy files; the system does not expose them. However, coach prescriptions manifest as changed villager behaviour in the public day debate (e.g. earlier Seer claims). That changed behaviour is recorded in `game_record` and consumed by `_update_strategy_wolf` the next game. The causal chain is

```
coach_text → villager_prompt → villager_policy → public_action_tokens
           → wolf_game_record → wolf_self_analysis → wolf_strategy_file
```

This is inverse-policy inference via behavioural observation. Any strategy the Coach can prescribe is a strategy with a publicly observable signature; if it had no public signature the Coach could not have written it, because the Coach itself only observes the public log. The paradigm is therefore **behaviourally transparent**: the set of coachable strategies equals the set of strategies adversaries can reverse-engineer from play.

## 6. LLM Technical Characterisation

The coach–player loop is a text-based episodic-memory system around frozen LLMs. No gradient updates occur. "Learning" is implemented as:

1. **Persistent strategy file** — plaintext appended to the system prompt via `_write_strategy` / `_build_setup_prompt` (`players/base_player.py:846–855`). This is non-parametric, soft memory.
2. **In-context reasoning** — at inference time the agent conditions on `system_prompt + strategy_file + dialogue_history` and emits the next action.

Transfer of coach knowledge is entirely textual. Coach → villager is a direct text insertion. Coach → wolf is indirect and observational: the wolf never reads coach text; it reads villager behaviour and reconstructs the underlying policy at 31B.

Architecturally this is in the same family as Reflexion, Voyager, and Generative-Agents designs: a frozen LLM with an external text memory updated by self-reflection. The present system's distinctive feature is that both sides run such a loop concurrently, and the two loops have asymmetric executor capacity (4B vs 31B).

The empirical consequence is that over N games the villager strategy files accumulate high-quality rules that the 4B executor only partially follows, while the wolf strategy files accumulate high-quality rules that the 31B executor follows faithfully and generalises beyond their literal wording. Gap widens monotonically. Coaching produces a measurable lift but on a plateau below parity.

## 7. Discussion

The result is consistent with the view that the coach–player paradigm is a **strategy-layer intervention** whose effectiveness is upper-bounded by execution-layer and rule-layer conditions. In the present configuration:

- The rule layer structurally favours wolves (win threshold, role self-exposure, coordination channel).
- The execution layer is capped by the 4B villager model's instruction-following and adversarial robustness.
- The paradigm layer is behaviourally transparent and bilaterally adaptive, so coaching shifts the opponent's adaptation target but does not keep pace with it.

None of this contradicts the project's hypothesis; it refines it. Coaching does improve villager performance relative to a hypothetical no-learning baseline. But coaching alone cannot produce parity.

## 8. Future Work

- **Equalise executor capacity.** Run `coach_and_self_analyze` with villager and wolf at matched model sizes to isolate paradigm contribution from capability gap.
- **Grant the Coach private logs.** Allow the Coach to read `_seer_log`, `_guard_log`, `_witch_log`, and `_wolf_debate_log`. This turns the Coach from a same-observability commentator into an information-privileged oracle, bringing it closer to a learning signal.
- **Introduce in-game coach interventions.** Move coaching from between-game strategy rewrites to per-round nudges that can respond to adversarial in-context manipulation.
- **Constrain wolf self-improvement.** Run an ablation in which wolf self-analysis is disabled, to measure how much of the residual gap is paradigm-inherent versus adversary-adaptation.
- **Add a villager-side communication channel.** Implement a restricted shared log for Seer / Guard / Witch to partially match `_wolf_debate_log`, allowing coordination the coach can reasonably prescribe.
- **Adversarial robustness training for villagers.** Instead of lexical rules, prescribe semantic-classifier behaviour that is more stable under paraphrase.

## 9. Conclusion

The coach–player paradigm, as implemented, is architecturally sound but empirically bounded. The 100-game `coach_and_self_analyze` run shows that a 31B Coach can produce high-quality, game-relevant prescriptions, but that the observed wolf dominance persists. The dominance is explained by a combination of rule-layer asymmetries that coaching cannot touch, a capability gap between the 4B villager executor and the 31B wolf executor, and a behaviourally transparent adversary that self-improves in parallel at higher capacity. The project's contribution is therefore not a refutation of the paradigm but a characterisation of its scope: coaching is necessary but not sufficient, and future gains require interventions at the capability and rule layers in addition to the strategy layer.

## Appendix A: File References

| File | Role |
|---|---|
| `game.py:101–120` | Win-condition evaluation |
| `game.py:141–172` | Wolf night debate loop |
| `game.py:87–93` | Log channel definitions |
| `players/wolf.py:28–62` | Wolf elimination prompt construction |
| `players/base_player.py:209–226` | Suspicion initialisation |
| `players/base_player.py:697–734` | Strategy-update dispatch |
| `players/base_player.py:737–796` | Villager strategy update |
| `players/base_player.py:798–844` | Wolf strategy update |
| `players/base_player.py:846–855` | Strategy persistence and prompt refresh |
| `config.py:31–32` | Model assignment |
| `config.py:38–63` | Scenario definitions |
| `run.py:244` | Coach instantiation with wolf model |
