"""Standalone diagnostic test for ElevenLabs Turbo TTS."""

import time
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

if not API_KEY:
    print("\n[ERROR] ELEVENLABS_API_KEY is not set in your .env file!")
    print("Add it like this: ELEVENLABS_API_KEY=your_key_here\n")
    exit(1)

try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs import play
except ImportError:
    print("\n[ERROR] Missing elevenlabs package. Install it via:")
    print("pip install elevenlabs\n")
    exit(1)

client = ElevenLabs(api_key=API_KEY)

TEST_PHRASES = [
    "Illumin system initialized. Listening for navigation targets.",
    "Target found: signature line. Move your hand two inches forward.",
    "Target reached. You are directly on the line.",
]

print("\n--- ElevenLabs Diagnostic Suite ---")
print(f"API Key detected: {API_KEY[:6]}...{API_KEY[-4:]}\n")

for i, phrase in enumerate(TEST_PHRASES, 1):
    print(f"[{i}/{len(TEST_PHRASES)}] Testing phrase: \"{phrase}\"")
    t_start = time.perf_counter()
    try:
        audio_stream = client.text_to_speech.convert(
            text=phrase,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_turbo_v2_5",
        )
        latency = (time.perf_counter() - t_start) * 1000.0
        print(f" -> Generation latency: {latency:.0f} ms. Playing audio...")
        play(audio_stream)
        print(" -> Playback finished.\n")
    except Exception as e:
        print(f" -> [FAIL] API call or playback failed: {e}\n")
        break

print("Diagnostic complete.")
