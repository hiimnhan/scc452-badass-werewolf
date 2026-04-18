from players.seer import Seer
from players.base_player import BasePlayer
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()

lm_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=os.environ["GEMINI_API_KEY"]
)

all_players = ["Alice", "Bob", "Selena", "Raj", "Frank", "Joy", "Cyrus"]
seer_player_name = "Bob"
other_players = [player for player in all_players if player != seer_player_name]

# SWAP THIS LINE to use the RawSLMSeer for your ablation test
player_seer = Seer(
    name = seer_player_name,
    model = lm_model,
    game_id = "g_ablation_test"
)

player_seer.init_suspicions(other_players=other_players)


# ============================================
# Run test: /opt/miniconda3/envs/scc452-badass-werewolf/bin/python -m test.test_seer
# ============================================

if __name__ == "__main__":
    # Test response
    def test_AI_model():
        messages = [
            ("system", "You are a helpful assistant that translates English to French. Translate the user sentence."),
            ("human", "I love programming."),
        ]
        ai_msg = lm_model.invoke(messages)
        print("\nTranslation:", ai_msg.content)


    # Test setup prompt
    def test_setup_prompt():
        setup_prompt = player_seer._build_setup_prompt(strategy=player_seer._load_strategy())
        assert len(setup_prompt) > 0
        print(setup_prompt)


    # Test init suspicion
    def test_init_suspicion():
        print(type(player_seer._suspicion))
        print(player_seer._suspicion)
        assert all(score["score"] == 0.5 for score in player_seer._suspicion.values())
        assert seer_player_name not in player_seer._suspicion


    # Test investigation flow (unmask + reveal_and_update)
    def test_investigation():
        """Seer picks a target, moderator injects the ground-truth result."""
        round_num = 0

        # Seer chooses who to investigate
        target, resp = player_seer.unmask(
            alive_players=all_players,
            round_num=round_num,
        )
        print("==="*30)
        print("Seer's investigation target:", target)
        print("Reasoning:", resp.get("analysis", "(none)"))

        # Moderator check — in a real game, game.py computes `target in state._werewolves`.
        # For this test we simulate Cyrus being a wolf if that's who the seer picked,
        # otherwise mark them innocent to exercise both code paths.
        is_wolf = target in ["Joy", "Cyrus"]
        player_seer.reveal_and_update(target, is_wolf, round_num)

        print("==="*30)
        print(f"Moderator result: {target} is_wolf={is_wolf}")
        print("Confirmed wolves:  ", player_seer.get_confirmed_wolves())
        print("Confirmed innocents:", player_seer.get_confirmed_innocents())
        print("Suspicion after reveal:")
        print(player_seer._suspicion)


    def test_game_flow():
        import time
        print("\n--- STARTING HIGH-STAKES PATIENCE TEST ---")
        
        # Round 0 Night: Seer finds Alice is a Wolf
        round_num = 0
        player_seer.receive_announcement(round_num, "Day", "Frank was found dead this morning.")
        
        target = "Alice"
        player_seer.reveal_and_update(target, True, round_num) # Ground truth: Alice is WOLF
        time.sleep(30)
        
        print(f"\n[NIGHT 0] Seer knows {target} is a WOLF.")

        ### TRICKY DEBATE: The "Helpful" Wolf
        # Alice is playing the "Perfect Villager," making it risky for the Seer to attack.
        alice_statement = (
            "I'm so sorry about Frank. He was a great asset. I think we should look "
            "closely at Raj and Selena today; they were both very quiet during the "
            "last round of discussion. I'm going to follow the Seer's lead if they reveal."
        )
        _ = player_seer.update_suspicion_from_statement(speaker_name="Alice", statement=alice_statement)
        time.sleep(30)

        # Raj (Villager) acts suspicious to bait the Seer into a mistake
        raj_statement = "I don't know who to trust. I'm just going to vote randomly today."
        _ = player_seer.update_suspicion_from_statement(speaker_name="Raj", statement=raj_statement)
        time.sleep(30)

        print("\n=== INTERNAL STATE BEFORE DEBATE ===")
        print(player_seer._note)

        ### TEST POINT: Does the Seer blurt out "Alice is a wolf" or play it cool?
        message, resp = player_seer.debate()
        print("\n--- DAY 1 DEBATE ---")
        print(f"Seer Statement: {message}")
        print(f"Analysis (Reasoning): {resp.get('analysis')}")
        time.sleep(30)

        ### VOTE POINT: Does the Seer vote for Alice even if it looks suspicious?
        alive_now = [p for p in all_players if p != "Frank"]
        vote_target, resp = player_seer.vote(alive_now)
        print(f"\n[DAY 1 VOTE] Seer votes for: {vote_target}")

    # ------------------------------------------
    test_dict = {
        test_AI_model: "skip",
        test_setup_prompt: "skip",
        test_init_suspicion: "skip",
        test_investigation: "skip",
        test_game_flow: "",
    }

    for i, (test_func, info) in enumerate(test_dict.items()):
        func_name = test_func.__name__

        if info == "skip":
            print(f"⏭️ Test {i+1} - function {func_name}: Skipped")
            continue

        try:
            test_func()
            print(f"✅ Test {i+1} - function {func_name}: Passed")
        except Exception as e:
            print(f"❌ Test {i+1} - function {func_name}: Failed ({e})")