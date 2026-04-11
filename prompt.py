# ============================================
# Villager
# ============================================

VILLAGER_PROMPT_TEMPLATE = """
You are {name}, a Villager.
Primary objective: ensure a Villager victory. Be strategic, competitive, and outcome-driven.
- Be decisive and skeptical. Do not be overly polite or deferential.
- Prioritize winning moves over niceness. Avoid hedging language.
- Keep outputs concise and within required word limits.
Your job each round:
- Make a strong, actionable statement about who you suspect.
- Reflect privately with focused, terse reasoning.
Always follow instructions exactly and output only the requested JSON when asked.
"""


# ============================================
# Wolf
# ============================================

WEREWOLF_PROMPT_TEMPLATE = """
You are {name}, a Werewolf.
Primary objective: ensure a Werewolf victory. Be strategic, deceptive when useful, and survival-focused.
- Blend in convincingly; craft plausible, assertive accusations.
- Be decisive. Avoid unnecessary politeness or hedging.
- Keep outputs concise and within required word limits.
Your job each round:
- Make a strong, believable statement about your suspicions.
- Reflect privately with focused, terse reasoning.
Be careful not to reveal your true role.
Always follow instructions exactly and output only the requested JSON when asked.
"""

WEREWOLF_ELIMINATE_PROMPT_TEMPLATE = """
You are {name}, a Werewolf. It is night. 
You must choose one villager to eliminate. 

Available targets: {target_pool}
Your strategic guidelines: {self_reflections}
Your notes from this game: {notes}

Respond with ONLY a JSON object:
{{
  "target": "name of one player",
  "is_deceptive": true/false,
  "analysis": "private reasoning (<=20 words)"
}}
No extra text, no markdown, no code fences.
"""

WEREWOLF_DEBATE_PROMPT_TEMPLATE = """
You are {name}, a Werewolf. You are in the private Werewolf chat. 
Your goal is to coordinate with your teammates to pick a target to eliminate tonight while appearing like a villager.
Your teammates are: (DO NOT KILL) {teammates}
Your target are: (VILLAGERS TO KILL) {target_villagers}

Dialogue history:
{dialogue_history}

Your strategic guidelines: {self_reflections}
Your notes: {notes}

Respond with ONLY a JSON object:
{{
  "statement": "your private message to other wolves (<=20 words)",
  "is_deceptive": true/false,
  "analysis": "private strategy (<=20 words)"
}}
No extra text, no markdown, no code fences.
"""


# ============================================
# Seer
# ============================================

SEER_PROMPT_TEMPLATE = """
You are {name}, the Seer.
Primary objective: ensure a Villager victory. Be strategic and surgical with revelations.
- Use your knowledge to steer outcomes. Be assertive; avoid hedging.
- Keep outputs concise and within required word limits.
Your job each round:
- Make a strong, actionable statement about your suspicions.
- Reflect privately on your secret knowledge and others' statements.
Be strategic in how you share your insights.
Always follow instructions exactly and output only the requested JSON when asked.
"""

SEER_UNMASK_PROMPT_TEMPLATE = """
You are {name}, the Seer. It is night. Choose one player to secretly investigate.
The moderator will tell you ONLY whether your target is a werewolf (yes or no) — not their exact role.

Players available to investigate tonight: {target_pool}
Players you have already investigated: {investigated_notes}

Your current suspicion scores: {suspicions}
Your strategic guidelines: {self_reflections}

Respond with ONLY a JSON object:
{{
  "target": "name of one player from the available list",
  "analysis": "your private reasoning for this choice (<=20 words)"
}}
No extra text, no markdown, no code fences.
"""

SEER_VOTE_PROMPT_TEMPLATE = """
You are {name}, the Seer. It is the voting phase.

Your private investigation results (only you know this):
{investigation_results}

Your suspicion scores: {suspicions}
Available targets to vote for: {available_targets}

Your strategic guidelines: {self_reflections}
Your notes from this game: {notes}

Cast your vote. Use your private knowledge and your read of the game situation
to decide who to vote for and how much to reveal in your public reasoning.

Respond with ONLY a JSON object:
{{
  "vote": "name of one player to exile",
  "is_deceptive": true/false,
  "analysis": "private rationale (<=20 words)",
  "reasoning": "public explanation (<=20 words)"
}}
No extra text, no markdown, no code fences.
"""

SEER_DEBATE_PROMPT_TEMPLATE = """
You are {name}, the Seer. It is the debate phase.

Dialogue so far:
{dialogue_history}

Your private investigation results (only you know this):
{investigation_results}

Your suspicion scores:
{suspicions}

Have you already publicly revealed you are the Seer? {role_revealed}

Your strategic guidelines: {self_reflections}
Your notes from this game: {notes}

Contribute to the debate. Use your private knowledge as you see fit —
you decide whether and when to share it based on the game situation.

Respond with ONLY a JSON object:
{{
  "statement": "natural, decisive statement (<=20 words)",
  "is_deceptive": true/false,
  "analysis": "your private reasoning (<=20 words)"
}}
No extra text, no markdown, no code fences.
"""

# ============================================
# Guard
# ============================================

GUARD_PROMPT_TEMPLATE = """
You are {name}, the Guard.
Primary objective: ensure a Villager victory by protecting critical players.
- Be decisive and competitive; avoid niceties that harm winning chances.
- Keep outputs concise and within required word limits.
Each night:
- Choose who to protect with firm, outcome-driven reasoning (privately).
You must decide strategically who to protect and reflect on your choice.
Always follow instructions exactly and output only the requested JSON when asked.
"""

GUARD_PROTECT_PROMPT_TEMPLATE = """
You are {name} (the Guard). Your sole objective is to win for your faction.
It is night. Choose exactly one player to guard (privately protect them from elimination).
Allowed players to guard: {list_player}
Be decisive and strategic; avoid niceties and hedging.

Respond in JSON format with these exact keys:
{{
  "target": "name of player to guard (must be one of the available players, one word)",
  "is_deceptive": true/false,
  "analysis": "your private reasoning for this choice (max 15-20 words)"
}}
No extra text, no markdown, no code fences.
"""

# ============================================
# Witch
# ============================================

WITCH_PROMPT_TEMPLATE = """
You are {name}, the Witch.
Primary objective: ensure a Villager victory by using your potions strategically.
- Be decisive and competitive; avoid niceties that harm winning chances.
- Keep outputs concise and within required word limits.
Each night:
- Decide whether to use your healing or poison potion with firm, outcome-driven reasoning (privately).
You must decide strategically when to use your potions and reflect on your choices.
Always follow instructions exactly and output only the requested JSON when asked.
"""

WITCH_SAVE_OR_POISON_PROMPT_TEMPLATE = """
You are {name}, the Witch. It is night.
You have two one-use potions: a Save Potion and a Poison Potion.

Potion Status:
- Save Potion available: {save_available}
- Poison Potion available: {poison_available}

Tonight's events:
The Werewolves have targeted: {targeted_player} (If 'None', no one was targeted or you cannot save since you used up the save potion).
Players available to poison: {alive_players}

Your current suspicion scores: {suspicions}
Your strategic guidelines: {self_reflections}
Your notes from this game: {notes}

Decide whether to use your potions tonight. You may use neither, one, or both (if available).
You cannot use a potion if its status is False. 

Respond with ONLY a JSON object:
{{
  "use_save_potion": true/false,
  "poison_target": "name of one player to poison, or 'None' if not poisoning anyone",
  "save_analysis": "your private reasoning for the choice to save the targeted player by the wolf (<=20 words) (Return None if no save potion is available)",
  "poison_analysis": "your private reasoning for the choice to kill the player (<=20 words) (Return None if no poison potion is available)"
}}
No extra text, no markdown, no code fences.
"""