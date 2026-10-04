"""PC side of the pen: reads ImuSamples / ImuEvents from the Arduino over serial."""
from __future__ import annotations

import glob
import queue
import threading
import time
from collections import deque
from typing import List, Optional

import serial

from src.config import config
from src.contracts import (
    DEVICE_BAUD, Clock, HapticCmd, ImuEvent, ImuSample, MotionState, RealClock,
    format_haptic_line, parse_device_line,
)


def find_port() -> str:
    ports = sorted(glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/ttyACM*"))
    return ports[0] if ports else config.ARDUINO_PORT


class ArduinoImuLink:
    def __init__(self, clock: Optional[Clock] = None, port: Optional[str] = None) -> None:
        self.clock = clock or RealClock()
        self.port = port
        self.motion = MotionState.STILL
        self._events: "queue.Queue[ImuEvent]" = queue.Queue()
        self._samples: deque = deque(maxlen=1000)
        self._ser: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self.send(HapticCmd.OFF)
        with self._lock:
            if self._ser:
                self._ser.close()
                self._ser = None

    def poll(self) -> List[ImuEvent]:
        out = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                return out

    def send(self, cmd: HapticCmd) -> None:
        with self._lock:
            if not self._ser:
                return
            try:
                self._ser.write(format_haptic_line(cmd).encode())
            except serial.SerialException:
                pass

    def recent_samples(self, n: int = 200) -> List[ImuSample]:
        return list(self._samples)[-n:]

    def _connect(self) -> Optional[serial.Serial]:
        port = self.port or find_port()
        try:
            ser = serial.Serial(port, DEVICE_BAUD, timeout=0.1)
            print(f"[ArduinoImuLink] Connected on {port}")
            return ser
        except serial.SerialException as e:
            print(f"[ArduinoImuLink] Cannot open {port}: {e}. Retrying...")
            return None

    def _reader(self) -> None:
        while self._running:
            if self._ser is None:
                ser = self._connect()
                if ser is None:
                    time.sleep(1.0)
                    continue
                with self._lock:
                    self._ser = ser
            try:
                raw = self._ser.readline()
            except (serial.SerialException, OSError, TypeError, AttributeError):
                if not self._running:
                    return
                print("[ArduinoImuLink] Lost connection. Reconnecting...")
                with self._lock:
                    if self._ser:
                        self._ser.close()
                    self._ser = None
                continue
            if not raw:
                continue
            msg = parse_device_line(raw.decode(errors="ignore"), self.clock.now())
            if isinstance(msg, ImuSample):
                self._samples.append(msg)
            elif isinstance(msg, ImuEvent):
                if msg.motion is not None:
                    self.motion = msg.motion
                self._events.put(msg)
