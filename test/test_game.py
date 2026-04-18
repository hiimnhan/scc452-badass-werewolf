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
# Run test: /opt/miniconda3/envs/scc452-badass-werewolf/bin/python -m test.test_game
# ============================================

if __name__ == "__main__":
    

    
    # ------------------------------------------
    test_dict = {
        
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