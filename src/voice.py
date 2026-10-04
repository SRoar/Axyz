"""VoiceEngine: streaming ElevenLabs TTS with macOS say fallback. (Background STT disabled.)"""

import os
import queue
import subprocess
import tempfile
import threading
from typing import Dict, Optional, Tuple
# Voice commands disabled: pen taps drive the flow now.
# import speech_recognition as sr
from src.config import config
from src.contracts import Clock, RealClock

try:
    from elevenlabs.client import ElevenLabs
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False

try:
    import pyaudio
except ImportError:
    pyaudio = None


class VoiceEngine:
    VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
    MODEL_ID = "eleven_flash_v2_5"
    SAMPLE_RATE = 22050
    OUTPUT_FORMAT = "pcm_22050"     # raw 16-bit mono PCM, played as it streams in
    WRITE_BYTES = 2048              # ~45 ms of audio per write, so silence() reacts quickly
    REQUEST_TIMEOUT_S = 5

    def __init__(self, clock: Optional[Clock] = None):
        self.clock = clock or RealClock()
        self.command_queue: "queue.Queue[str]" = queue.Queue()
        self.tts_queue: "queue.Queue[Tuple[int, str]]" = queue.Queue()
        # self.recognizer = sr.Recognizer()
        self.stop_listening = None

        self._running = True
        self._lock = threading.Lock()
        self._pending = 0           # phrases queued or playing; is_speaking() == pending > 0
        self._generation = 0        # bumped by silence(); stale phrases are dropped
        self._current_process: Optional[subprocess.Popen] = None
        self._cache: Dict[str, bytes] = {}
        self._pa = pyaudio.PyAudio() if pyaudio else None

        self.client = None
        if ELEVENLABS_AVAILABLE and config.ELEVENLABS_API_KEY:
            try:
                self.client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)
                print("[VoiceEngine] ElevenLabs TTS initialized.")
            except Exception as e:
                print(f"[VoiceEngine] ElevenLabs initialization error: {e}")
        if not self.client:
            print("[VoiceEngine] ElevenLabs unavailable. Using macOS 'say' fallback.")

        self._start_tts_worker()
        # self._start_stt_listener()

    # def _start_stt_listener(self):
    #     try:
    #         mic = sr.Microphone()
    #         with mic as source:
    #             self.recognizer.adjust_for_ambient_noise(source, duration=0.8)
    #
    #         def audio_callback(recognizer, audio):
    #             if self.is_speaking():
    #                 return
    #             try:
    #                 text = recognizer.recognize_google(audio).lower()
    #                 print(f"\n[VoiceEngine] Recognized speech: '{text}'")
    #                 self.command_queue.put(text)
    #             except (sr.UnknownValueError, sr.RequestError):
    #                 pass
    #
    #         self.stop_listening = self.recognizer.listen_in_background(mic, audio_callback)
    #         print("[VoiceEngine] Background microphone listener active.")
    #     except Exception as e:
    #         print(f"[VoiceEngine] Mic warning (voice commands disabled): {e}")

    def _cancelled(self, gen: int) -> bool:
        return gen != self._generation or not self._running

    def _write_pcm(self, stream, data: bytes, gen: int) -> bool:
        for i in range(0, len(data), self.WRITE_BYTES):
            if self._cancelled(gen):
                return False
            stream.write(data[i:i + self.WRITE_BYTES])
        return True

    def _play_elevenlabs(self, phrase: str, gen: int) -> bool:
        if not self.client:
            return False
        if not self._pa:
            return self._play_elevenlabs_afplay(phrase, gen)
        played_any = False
        stream = None
        try:
            stream = self._pa.open(format=pyaudio.paInt16, channels=1,
                                   rate=self.SAMPLE_RATE, output=True)
            cached = self._cache.get(phrase)
            if cached is not None:
                self._write_pcm(stream, cached, gen)
                return True

            audio = bytearray()
            carry = b""
            for chunk in self.client.text_to_speech.stream(
                self.VOICE_ID,
                text=phrase,
                model_id=self.MODEL_ID,
                output_format=self.OUTPUT_FORMAT,
                request_options={"timeout_in_seconds": self.REQUEST_TIMEOUT_S},
            ):
                if not chunk:
                    continue
                audio += chunk
                data = carry + chunk
                even = len(data) - (len(data) % 2)   # never split a 16-bit sample
                data, carry = data[:even], data[even:]
                played_any = True
                if not self._write_pcm(stream, data, gen):
                    return True
            self._cache[phrase] = bytes(audio)
            return True
        except Exception as e:
            print(f"[VoiceEngine] ElevenLabs TTS error: {e}")
            return played_any
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()

    def _play_elevenlabs_afplay(self, phrase: str, gen: int) -> bool:
        """No pyaudio: fetch the whole MP3, then play it with macOS afplay."""
        path = None
        try:
            audio = self.client.text_to_speech.convert(
                self.VOICE_ID,
                text=phrase,
                model_id=self.MODEL_ID,
                request_options={"timeout_in_seconds": self.REQUEST_TIMEOUT_S},
            )
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                path = f.name
                for chunk in audio:
                    f.write(chunk)
            self._run_process(["afplay", path], gen)
            return True
        except Exception as e:
            print(f"[VoiceEngine] ElevenLabs TTS error: {e}")
            return False
        finally:
            if path and os.path.exists(path):
                os.remove(path)

    def _run_process(self, cmd, gen: int):
        try:
            with self._lock:
                if self._cancelled(gen):
                    return
                proc = subprocess.Popen(cmd)
                self._current_process = proc
            proc.wait()
        finally:
            with self._lock:
                self._current_process = None

    def _play_say_fallback(self, phrase: str, gen: int):
        try:
            self._run_process(["say", phrase], gen)
        except Exception as e:
            print(f"[VoiceEngine] System 'say' failed: {e}")

    def _start_tts_worker(self):
        def worker():
            while self._running:
                try:
                    gen, phrase = self.tts_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if not self._cancelled(gen):
                        print(f"[VoiceEngine Playing]: \"{phrase}\"")
                        if not self._play_elevenlabs(phrase, gen):
                            self._play_say_fallback(phrase, gen)
                finally:
                    with self._lock:
                        self._pending -= 1

        threading.Thread(target=worker, daemon=True).start()

    def speak(self, text: str, interrupt: bool = False):
        if interrupt:
            self.silence()
        with self._lock:
            self._pending += 1
            gen = self._generation
        self.tts_queue.put((gen, text))

    def silence(self):
        with self._lock:
            self._generation += 1
            while True:
                try:
                    self.tts_queue.get_nowait()
                    self._pending -= 1
                except queue.Empty:
                    break
            if self._current_process and self._current_process.poll() is None:
                self._current_process.terminate()

    def is_speaking(self) -> bool:
        with self._lock:
            return self._pending > 0

    def poll_command(self) -> Optional[str]:
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.silence()
        self._running = False
        if self.stop_listening:
            try:
                self.stop_listening(wait_for_stop=False)
            except Exception:
                pass
