"""Standalone test for microphone speech recognition + ElevenLabs reply."""

import os
import subprocess
import tempfile
import speech_recognition as sr
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()

ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY", "")
if not ELEVEN_KEY:
    print("[ERROR] ELEVENLABS_API_KEY is not set in your .env file!")
    exit(1)

tts_client = ElevenLabs(api_key=ELEVEN_KEY)


def speak(text: str):
    """Speaks text using ElevenLabs Turbo played over native macOS afplay."""
    print(f"\n[Illumin Speaking]: \"{text}\"")
    try:
        audio_gen = tts_client.text_to_speech.convert(
            text=text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_turbo_v2_5",
        )
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            for chunk in audio_gen:
                f.write(chunk)
            temp_path = f.name
        subprocess.run(["afplay", temp_path], check=False)
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except Exception as e:
        print(f"[TTS Error]: {e}")


def main():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("\n--- Audio Diagnostic Test ---")
    print("Calibrating microphone for ambient background noise...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1.0)

    speak("Voice system ready. Tell me what you would like to find.")

    print("\n[LISTENING] Speak clearly into your mic (e.g. 'find signature', 'where do I sign', 'find date')...")
    print("Press Ctrl+C to stop.\n")

    while True:
        try:
            with mic as source:
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
            
            print("[Processing speech...]")
            text = recognizer.recognize_google(audio).lower()
            print(f">>> You said: \"{text}\"")

            # Check matching logic
            if any(k in text for k in ["signature", "sign"]):
                speak(f"Understood. Locating the signature line.")
            elif any(k in text for k in ["date", "day"]):
                speak(f"Understood. Locating the date field.")
            elif "hello" in text or "hi" in text:
                speak("Hello! I am ready to guide your hand.")
            else:
                speak(f"I heard you say: {text}. That target is not recognized yet.")

        except sr.WaitTimeoutError:
            print("Listening timed out waiting for phrase, still listening...")
        except sr.UnknownValueError:
            print("Could not understand audio. Try speaking closer to the mic.")
        except KeyboardInterrupt:
            print("\nExiting voice test.")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
