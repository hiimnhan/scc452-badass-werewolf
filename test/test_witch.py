from players.witch import Witch
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
witch_player_name = "Joy"
other_players = [player for player in all_players if player != witch_player_name]

player_witch = Witch(
    name = witch_player_name,
    model = lm_model,
    game_id = "g042"
)

player_witch.init_suspicions(other_players=other_players)
player_witch._save_available = True
player_witch._poison_available = True


# ============================================
# Run test: /opt/miniconda3/envs/scc452-badass-werewolf/bin/python -m test.test_witch
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
        setup_prompt = player_witch._build_setup_prompt(strategy=player_witch._load_strategy())
        assert len(setup_prompt) > 0
        print(setup_prompt)
    
    
    # Test init suspicion
    def test_init_suspicion():
        print(type(player_witch._suspicion))
        print(player_witch._suspicion)
        assert all(score["score"] == 0.5 for score in player_witch._suspicion.values())
        assert witch_player_name not in player_witch._suspicion
        
    
    # Test game summary
    def test_game_summary():
        """Witch correctly accumulates night and day announcements in _game_summary."""
        player_witch.receive_announcement(0, "Night", "Frank was targeted by wolves.")
        actions, resp = player_witch.save_or_poison(
            targeted_player_by_wolves="Frank",
            alive_players=all_players,
            round_num=0,
        )
        print(player_witch._game_summary)
        
    def test_game_flow():
        import time
        # Round 0
        round_num = 0
        player_witch.receive_announcement(round_num, "Night", "Frank was targeted by wolves.")
        player_witch.receive_announcement(round_num, "Day", "No one was exiled.")
        
        ### Get bid
        bid, resp = player_witch.get_bid()
        print("==="*30)
        print("Player's bid:", bid)
        time.sleep(30)
        
        _ = player_witch.update_suspicion(speaker_name="Bob", statement="I am a Seer. I looked into Selena and Raj and see that they are both wolves. We need to vote them out.")
        
        ### Update suspicion after a statement
        print("==="*30)
        print("Updated suspicion:")
        print(player_witch._suspicion)
        time.sleep(30)
        
        ### Debate: As everything is updated in the note, do we need the dialouge history?
        message, _ = player_witch.debate()
        print("==="*30)
        print("Player's statement:")
        print(message)
        time.sleep(30)
        
        ### Vote
        vote_target, resp = player_witch.vote(all_players)
        print("==="*30)
        print("Player's vote response:", vote_target)
        print(resp)
        time.sleep(30)
        
        # Round 1 Night
        round_num = 1
        player_witch.receive_announcement(round_num, "Night", "Bob was targeted by wolves.")
        
        actions, resp = player_witch.save_or_poison(
            targeted_player_by_wolves="Bob",
            alive_players=all_players,
            round_num=round_num,
        )
        print("==="*30)
        print("Player's note:")
        print(player_witch._note)

    
    # ------------------------------------------
    test_dict = {
        test_AI_model: "skip", 
        test_setup_prompt: "skip",
        test_init_suspicion: "skip",
        test_game_summary: "skip",
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