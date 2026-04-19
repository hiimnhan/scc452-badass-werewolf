from players.wolf import Wolf
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Initialize Model
lm_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", temperature=0.7, google_api_key=os.environ["GEMINI_API_KEY"]
)

# Setup Test Data
all_players = ["Alice", "Bob", "Selena", "Raj", "Frank", "Joy", "Cyrus"]
wolf_name = "Cyrus"
other_wolves = ["Selena"]
villagers = [p for p in all_players if p not in other_wolves and p != wolf_name]

player_wolf = Wolf(name=wolf_name, model=lm_model, game_id="test_v1")

player_wolf.init_suspicions(other_players=[p for p in all_players if p != wolf_name])


def test_wolf_danger_update():
    """Verify that suspicion updates treat villagers as 'Danger' sources."""
    print("\n" + "=" * 10 + " TESTING DANGER SCORE UPDATE " + "=" * 10)
    statement = "I am the Seer and I know Cyrus is a wolf!"
    resp = player_wolf.update_suspicion_from_statement(speaker_name="Alice", statement=statement)

    # DEBUG: ADD THIS LINE TO SEE THE RAW TEXT
    print(f"Raw AI Response: {resp.get('_raw_response')}")

    print(f"Analysis for Alice's threat: {resp.get('updates', {}).get('Alice')}")
    print(f"Current Danger Table:\n{player_wolf._format_suspicion_block()}")


def test_wolf_debate_flow():
    """Verify the 2-round internal wolf coordination."""
    print("\n" + "=" * 10 + " TESTING WOLF DEBATE " + "=" * 10)
    # Mock some prior dialogue
    mock_history = [
        ["Selena", "I think we should target the quiet one, Raj."],
    ]

    statement, log = player_wolf.wolf_debate(
        alive_players=all_players, other_wolves=other_wolves, dialogue_history=mock_history, round_num=1
    )

    print(f"Wolf Statement: {statement}")
    print(f"Internal Metadata: {log.get('analysis')}")


def test_eliminate_action():
    """Verify the final kill decision logic."""
    print("\n" + "=" * 10 + " TESTING ELIMINATION " + "=" * 10)
    target, log = player_wolf.eliminate(alive_players=all_players, round_num=1)

    print(f"Final Target Picked: {target}")
    print(f"Elimination Reasoning: {log.get('analysis')}")


def test_compiled_note_structure():
    """Verify the {note} block consolidation."""
    print("\n" + "=" * 10 + " TESTING NOTE CONSOLIDATION " + "=" * 10)
    # Add a public event
    player_wolf.receive_announcement(1, "Day", "Frank was exiled.")

    print("Full Working Note for LLM Context:")
    print("-" * 20)
    print(player_wolf._note)


if __name__ == "__main__":
    tests = [test_wolf_danger_update, test_wolf_debate_flow, test_eliminate_action, test_compiled_note_structure]

    for i, test in enumerate(tests):
        try:
            test()
            print(f"\n✅ Test {i + 1} ({test.__name__}): PASSED")
            time.sleep(2)  # Avoid rate limits if on free tier
        except Exception as e:
            print(f"\n❌ Test {i + 1} ({test.__name__}): FAILED\nError: {e}")
