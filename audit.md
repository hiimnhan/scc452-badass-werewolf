# Architectural Audit & Strategic Optimization: Werewolf LLM Coaching System

Before diving in, a framing note: your existing `report.md` already does excellent diagnostic work identifying rule-layer, capability-layer, and paradigm-layer asymmetries. I will not re-litigate those. Instead, I will focus on what is actionable *within your stated constraints* — prompt orchestration and multi-agent protocol design — and treat the 4B/31B capability gap as a fixed boundary condition rather than a grievance.

---

## 1. Failure Mode Analysis

Your feedback loop has three distinct failure sites. They are not equally severe, and conflating them has probably cost you diagnostic clarity.

### 1.1 The dominant failure is **comprehension-at-execution-time**, not feedback quality

The 31B Coach is producing perfectly serviceable prose. I can infer this from `coach.py` — the prompts are well-scoped, the two-step design (self-update → feedback) is clean, and you're using the 31B model for coach output. The feedback file, once written, is not the bottleneck.

The bottleneck is where that feedback *lands*. Examine `base_player.py:_build_setup_prompt`:

```python
parts = [
    self._role_definition.format(name=self._name) if self._role_definition else "",
    f"Your personality: {self._personality}" if self._personality else "",
    f"Your strategy from previous games:\n{strategy}" if strategy else "",
    GAME_RULES,
]
```

The strategy text arrives as a wall of prose appended to the system prompt. Your `_update_strategy_villager` prompt asks for `"strategy": "updated rules as bullet points (<=300 words)"`. So the 4B Gemma receives, on every call, an undifferentiated ~300-word bullet list alongside role definition, personality, 700-word game rules, and then a 500+ token human message containing game summary and suspicion scores.

A 4B model's *effective* attention window — the span over which it maintains coherent instruction-following under adversarial pressure — is much smaller than its *nominal* context window. Research on instruction-following degradation in small models (the "lost in the middle" phenomenon, amplified at smaller scales) tells us the villager is likely treating most of the strategy text as soft background, not hard constraint.

**Diagnostic test to confirm**: Grep your game logs for any case where the strategy file contains a rule like "never reveal Seer identity on Day 1" and the Seer revealed on Day 1 anyway. I'd wager you'll find dozens. That's not a feedback quality problem — the rule is clear. It's an attention/compliance problem at 4B scale.

### 1.2 The secondary failure is **feedback abstraction-level mismatch**

Your Coach prompt asks for feedback "using the roles of the players, NOT their names." This is correct for generalization, but it produces feedback pitched at a level of abstraction the 4B executor struggles to operationalize. Something like "The Seer revealed too early and was killed Night 2" becomes, in the strategy file, "Seers should delay reveal until evidence is strong." At inference time, the 4B Seer has no robust procedure for evaluating "evidence is strong" — it's a judgment primitive that requires capabilities the 4B model doesn't reliably have.

### 1.3 The tertiary failure is **retrieval/salience at decision time**

When the villager is casting a vote, the full strategy sits in the system prompt but there is no mechanism to surface *which rules apply to this decision*. The strategy is always fully present and therefore always partially ignored. A 4B model processing a vote decision isn't running a lookup against its strategy — it's pattern-matching on the immediate debate transcript.

### 1.4 Ranking these

| Failure site | Severity | Fixable via prompt orchestration? |
|---|---|---|
| Execution comprehension at 4B | High | Partially — via structure, not prose |
| Feedback abstraction mismatch | High | Yes — via structured directives (§2) |
| Retrieval/salience at decision | Medium | Yes — via gated memory (§3) |
| Feedback prose quality | Low | Already adequate |

The action items below target the two "yes, fixable" rows.

---

## 2. Feedback-to-Action Mapping: Structured Strategic Directives

Prose is the wrong representation for a 4B executor. Prose requires interpretation; interpretation requires capability you don't have. Replace it with a typed, decision-local directive schema.

### 2.1 The core insight

A 4B model is much better at JSON-template compliance than at prose-rule compliance. If a rule is phrased as "when X, do Y," it has to be evaluated. If a rule is phrased as a lookup table keyed on decision type and situation, it can be retrieved.

Restructure the Coach's output as a **Strategic Directive Set (SDS)** — a set of decision-local, role-indexed, conditionally-gated rules.

### 2.2 Proposed schema

```json
{
  "role": "Seer",
  "directives": [
    {
      "id": "SEER-001",
      "phase": "DEBATE",
      "round_condition": "round_num <= 2",
      "trigger": "considering revealing role",
      "action": "DO_NOT_REVEAL",
      "rationale_short": "Round 1-2 reveals died 87% of observed games",
      "confidence": 0.9,
      "supersedes": []
    },
    {
      "id": "SEER-002",
      "phase": "DEBATE",
      "round_condition": "confirmed_wolves_count >= 1",
      "trigger": "at least one confirmed wolf, you have not been attacked today",
      "action": "REVEAL_AND_ACCUSE",
      "rationale_short": "Delayed reveal only valuable when evidence exists",
      "confidence": 0.8,
      "supersedes": ["SEER-001"]
    },
    {
      "id": "SEER-VOTE-001",
      "phase": "VOTE",
      "trigger": "a confirmed wolf is a vote target",
      "action": "VOTE_CONFIRMED_WOLF",
      "rationale_short": "Seer's vote on confirmed info is highest-EV action",
      "confidence": 1.0,
      "supersedes": []
    }
  ]
}
```

### 2.3 How the executor consumes this

At each decision point (debate/vote/night action), inject **only the directives matching the current `phase`** into the prompt. This is not a memory retrieval problem — you already know the phase; filter on it before the LLM ever sees the list.

Modify `base_player.py`'s prompt construction so that the player prompt includes a section like:

```
=== Active Directives for this decision ===
[only directives matching current phase, sorted by confidence desc]

Directive SEER-001: Do not reveal your role this round. (Confidence: 0.9)
Directive SEER-002: If a confirmed wolf exists and you haven't been attacked, reveal and accuse them. (Confidence: 0.8, supersedes SEER-001)
```

Three to six directives per decision is the target. A 4B model handles this much more reliably than 300 words of flowing prose.

### 2.4 Coach output contract change

Change `_generate_coach_feedback` in `coach.py` to emit two artifacts:

1. **Narrative feedback** (what you have now) — for human-readability and logs.
2. **Structured directives** — a JSON file (e.g., `coach_directives.json`) keyed by role, which `BasePlayer._load_strategy` parses.

Prompt the Coach explicitly:

> Each directive MUST be atomic — a single trigger, a single action, a single phase. Do NOT write compound rules. Do NOT write rules whose trigger requires subjective judgment (e.g., "when the debate feels suspicious"). Triggers must reference observable game state: round number, alive count, votes received, confirmed investigations, etc.

This forces the Coach to do the abstraction-lowering work that the 4B executor cannot do itself. You are relocating cognitive load from the weak executor to the strong coach — which is the whole point of the paradigm.

### 2.5 A concrete win this buys you

Your `report.md` notes that villager strategies say things like "Guard must protect the Seer" but Guard has no Seer identifier. Under the SDS schema, this rule would *fail validation at Coach emission time* because its trigger (`target == Seer`) references non-observable state from Guard's perspective. The Coach would be forced to emit an executable rule instead (e.g., "protect the player who received the most votes yesterday without being exiled"). The schema itself enforces executability.

---

## 3. Memory & State Management: The Reflection Ledger

Your current memory model is flat: one strategy file, fully present in every call. Replace it with a layered architecture with three distinct timescales.

### 3.1 Three-tier memory architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ Tier 1: STABLE IDENTITY (unchanging within a game)               │
│   - Role definition, game rules, personality                     │
│   - Stays in system prompt                                       │
├──────────────────────────────────────────────────────────────────┤
│ Tier 2: STRATEGIC DIRECTIVES (per-decision filtered)             │
│   - Coach-emitted SDS, filtered by current phase                 │
│   - Injected into human message, not system                      │
├──────────────────────────────────────────────────────────────────┤
│ Tier 3: REFLECTION LEDGER (per-turn working memory)              │
│   - "What I did last turn, what happened, what I'd do next"      │
│   - Single recency entry, overwritten each turn                  │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 The Reflection Ledger specifically

After every one of its own actions, the player writes a 2-3 line reflection entry in a fixed schema. This is *not* a narrative memory — it is an explicit bridge between the last decision and the next.

```json
{
  "last_action": "voted to exile Jin",
  "last_reasoning": "Highest suspicion, contradictory statements",
  "directive_applied": "VILL-VOTE-003",
  "observed_outcome": "Jin was exiled but was Villager",
  "update_for_next_turn": "directive VILL-VOTE-003 failed; suspicion score alone insufficient"
}
```

Then, at the start of the next decision, the player sees its own last-turn reflection *before* anything else. This is the mechanism Reflexion (Shinn et al., 2023) showed works on small models: you are not asking the model to retrieve from long memory, you are handing it a freshly-written self-instruction.

### 3.3 Why this works specifically for 4B

A 4B model has weak *retrieval* (pulling relevant rules from a long prompt) but reasonable *adjacency* (acting on information in the immediate preceding span). The Reflection Ledger converts a retrieval problem into an adjacency problem.

### 3.4 Implementation sketch (minimally invasive)

Add to `BasePlayer`:

```python
self._reflection_ledger: dict = {}  # single entry, overwritten each decision

def _write_reflection(self, action, reasoning, directive_id, expected_outcome):
    self._reflection_ledger = {
        "last_action": action,
        "last_reasoning": reasoning,
        "directive_applied": directive_id,
        "expected_outcome": expected_outcome,
        # observed_outcome filled in by receive_announcement
    }
```

Call it at the end of `vote()`, `debate()`, and each night action. Then prepend its contents to the next decision prompt. Single entry, ~150 tokens, always adjacent to the new question.

### 3.5 What NOT to do

Do not implement a long-form reflection history that accumulates across turns. At 4B scale, adding more context reduces performance past a surprisingly modest threshold. One entry, overwritten. Trust the compression.

---

## 4. Experimental Rigor: Metrics Beyond Win/Loss

Win rate over 100 games has catastrophically low statistical power for detecting modest strategic improvement. In a 7-player game with structural wolf advantage, the villager win rate might move from 15% to 22% — that's a real improvement and you'd need ~400 games to detect it at p<0.05. Propose these process-level metrics instead; they move an order of magnitude faster than the outcome metric.

### 4.1 Metric 1: Directive Compliance Rate (DCR)

**Definition**: Of all decision points where a directive was applicable, the fraction where the agent's action matched the directive.

```
DCR = (decisions_matching_applicable_directive) / (decisions_with_applicable_directive)
```

This directly measures whether the feedback loop is *landing*. If DCR is below ~0.6, your comprehension failure (§1.1) is dominant and no amount of better coaching will help. If DCR is above 0.8 but win rate isn't rising, your Coach is giving good advice on losing strategies and the problem is at the Coach level.

This metric is diagnostic in a way win rate can never be.

### 4.2 Metric 2: Information Efficiency Index (IEI)

**Definition**: For information-privileged villager roles (Seer, Witch), the ratio of *decisions that used private information* to *decisions where using private information was optimal*.

Concretely for the Seer:

```
IEI_seer = (days_seer_publicly_used_investigation_result) / 
           (days_seer_had_unused_confirmed_wolf_alive)
```

If a Seer has a confirmed wolf alive and doesn't push them in the debate, that's wasted information. Early games should have IEI near 0.3 (seers hiding too long or revealing to wrong targets); strategic evolution should push this toward 0.7+.

This metric isolates the specific failure your paradigm is trying to fix — the capability to use asymmetric information well — and does so independently of whether villagers happened to win.

### 4.3 Metric 3: Deception Resistance Rate (DRR)

**Definition**: When a wolf makes a false accusation against a villager, the fraction of subsequent debate statements and votes that do *not* bandwagon on the accusation.

```
DRR = 1 - (villager_votes_bandwagoning_wolf_accusation) / 
          (total_villager_votes_post_wolf_accusation)
```

This directly measures the 4B model's adversarial robustness to wolf manipulation — precisely the capability your §5.2.2 identifies as the execution-layer weakness. If coaching works, DRR should rise over games even without winrate moving.

### 4.4 Why these three together

- **DCR** tests whether feedback *reaches* behavior (compliance).
- **IEI** tests whether villagers *exploit their asymmetric advantages* (positive capability).
- **DRR** tests whether villagers *resist wolf manipulation* (adversarial capability).

A system that is genuinely learning should show movement on all three before it shows movement on win rate. If you run 30 games and see DCR climb but IEI and DRR flat, you know exactly what the Coach needs to emphasize next. That's what win rate alone will never tell you.

---

## 5. Ablation Study Suggestions

Your current four scenarios vary multiple things simultaneously. To attribute improvement causally to *feedback quality* (your actual hypothesis), I propose the following additional conditions. Run these as paired comparisons.

### 5.1 Ablation A: Feedback Quality vs. Feedback Volume

**Control**: "Noise coach" — the Coach reads the game but outputs generic, game-agnostic strategic platitudes ("vote carefully," "pay attention to contradictions"). Same word count, same structural format as the real Coach.

**Test**: The real Coach.

If the real Coach doesn't outperform the Noise Coach, your improvement is coming from the *ritual* of strategy injection, not its *content*. This is the single most important ablation to run. Based on prior work with small models, I'd put maybe 30% probability on the Noise Coach matching the real Coach to within noise on your current setup — which would be a devastating finding worth knowing.

### 5.2 Ablation B: Structured vs. Prose Feedback

**Control**: Current prose-based strategy updates.

**Test**: Strategic Directive Set (§2) in structured JSON.

Same Coach, same game records, different output format. If the SDS version outperforms prose, you have direct evidence that the bottleneck was representation, not content.

### 5.3 Ablation C: Frozen Wolf Control

**Control**: Wolves self-analyze every game (current system).

**Test**: Wolves are frozen — no strategy updates at all after game 1. Only villagers update.

This is critical. Your `report.md` identifies §5.3.1 as a major issue: the adversary also learns, so gains cancel. Running the "frozen wolves" ablation tells you the *one-sided* improvement rate, which is the paradigm's true ceiling under your constraints. If frozen-wolf villager win rate climbs meaningfully while current-scenario win rate doesn't, you have isolated the paradigm effect from the adversary-adaptation confound. This is publishable cleanly.

### 5.4 Ablation D: Coach Privilege

**Control**: Coach sees only public logs (current).

**Test**: Coach sees all private logs (`_wolf_debate_log`, role logs).

This addresses your report's §5.3.2. The test condition isn't a realistic deployment — it's a ceiling measurement. It tells you how much of the current underperformance is due to Coach input blindness vs. executor capability. If Privileged Coach barely outperforms Blind Coach, the villagers can't use the information anyway and the 4B ceiling is the real wall. If Privileged Coach is much better, your paradigm has room if you can engineer better coach inputs.

### 5.5 Ablation E: Capability-Matched Sanity Check

Run `coach_and_self_analyze` with villagers at 31B (the wolf model). Just once, ~30 games. This tells you, definitively, whether your paradigm works *at all* when executor capacity is not the bottleneck. If 31B villagers also fail to close the gap, then the paradigm itself is limited and no amount of prompt engineering will save it. If 31B villagers reach parity or better, then your paradigm is sound and the 4B gap is a pure capability story. Either answer is valuable — and you can't get it from any other experiment.

### 5.6 What this matrix proves

| Comparison | Isolates |
|---|---|
| Real Coach vs Noise Coach | Content quality matters |
| Structured vs Prose | Representation matters |
| Frozen Wolves vs Adaptive | One-sided paradigm ceiling |
| Privileged vs Blind Coach | Coach information bottleneck |
| 31B Villager vs 4B | Paradigm soundness at capability |

Five ablations, each with a single-variable interpretation. This is the rigor bar your current four-scenario design doesn't clear.

---

## Closing Synthesis

Your current system is architecturally coherent but strategically under-optimized in three specific, fixable ways: feedback is delivered at the wrong abstraction level for a 4B executor, memory is flat when it should be layered, and your metrics don't let you diagnose where the loop breaks. The fixes — Strategic Directive Sets, Reflection Ledger, process-level metrics — are all pure prompt orchestration and respect your no-fine-tuning constraint.

A realistic expected improvement from implementing sections 2 and 3 is on the order of +5-10 percentage points in villager win rate, with possibly larger movement on DCR/IEI/DRR. That's not parity — your `report.md` is correct that parity requires addressing the rule layer, and that's outside the paradigm's scope. But it's enough to cleanly separate "coaching does nothing" (null result) from "coaching has measurable but bounded effect" (your actual, publishable finding).

One final suggestion: when you write up the results, frame the contribution as **"characterizing the scope of natural-language coaching for asymmetric-capability multi-agent systems."** That's a harder result than "coaching works" and a more honest one than "coaching failed." The ablation matrix above is designed to let you make that characterization with quantitative backing rather than post-hoc speculation.