"""Non-blocking voice input and ElevenLabs TTS output using native macOS audio."""

import os
import queue
import subprocess
import tempfile
import threading
import speech_recognition as sr
from src.config import config

try:
    from elevenlabs.client import ElevenLabs
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False


class VoiceEngine:
    def __init__(self):
        self.command_queue: queue.Queue[str] = queue.Queue()
        self.tts_queue: queue.Queue[str] = queue.Queue()
        self.recognizer = sr.Recognizer()
        self.stop_listening = None

        if ELEVENLABS_AVAILABLE and config.ELEVENLABS_API_KEY:
            self.client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)
            self._start_tts_worker()
            print("[VoiceEngine] ElevenLabs TTS initialized.")
        else:
            self.client = None
            print("[VoiceEngine] ElevenLabs API key missing or package not installed.")

        self._start_stt_listener()

    def _start_stt_listener(self):
        """Listens continuously in the background for target commands."""
        try:
            mic = sr.Microphone()
            with mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.8)

            def audio_callback(recognizer, audio):
                try:
                    text = recognizer.recognize_google(audio).lower()
                    print(f"\n[VoiceEngine] Heard: '{text}'")
                    self.command_queue.put(text)
                except (sr.UnknownValueError, sr.RequestError):
                    pass

            self.stop_listening = self.recognizer.listen_in_background(mic, audio_callback)
            print("[VoiceEngine] Background microphone listener active.")
        except Exception as e:
            print(f"[VoiceEngine] Mic warning (voice commands disabled, keyboard still works): {e}")

    def _start_tts_worker(self):
        """Processes spoken audio output in a background thread without stalling the video feed."""
        def worker():
            while True:
                phrase = self.tts_queue.get()
                try:
                    audio_generator = self.client.text_to_speech.convert(
                        text=phrase,
                        voice_id="JBFqnCBsd6RMkjVDRZzb",
                        model_id="eleven_turbo_v2_5",
                    )
                    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                        for chunk in audio_generator:
                            f.write(chunk)
                        temp_path = f.name

                    # Native macOS audio player (zero dependencies)
                    subprocess.run(["afplay", temp_path], check=False)
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception as e:
                    print(f"[VoiceEngine] TTS error: {e}")
                finally:
                    self.tts_queue.task_done()

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def speak(self, text: str):
        if self.client:
            self.tts_queue.put(text)
        else:
            print(f"[Voice Simulated TTS]: {text}")

    def poll_command(self) -> str | None:
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None