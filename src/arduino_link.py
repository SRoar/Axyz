"""ArduinoImuLink: the PC side of the pen probe.  (contract: src.contracts.ImuLink, owner: Dev 1)

    link = ArduinoImuLink(clock)       # constructor fixed by the contract
    link.start(); events = link.poll(); link.send(HapticCmd.LOCK); link.stop()

A background thread owns the serial port: it finds the board, drains whatever the board buffered while
nobody was listening (seen: minutes of stale lines), parses the stream with contracts.parse_device_line,
and reconnects by itself if the board drops off USB. Nothing here blocks the caller:
  poll()           -> ImuEvents (motion changes, taps) since the last call, oldest first
  motion           -> latest motion state (STILL until the board says otherwise)
  send(cmd)        -> queued, written by the thread; dropped (not queued up) while disconnected
  recent_samples() -> last n ImuSamples for the HUD waveform

    python -m src.arduino_link            # live monitor for bring-up: prints events and a 1 Hz status line
"""
from __future__ import annotations

import argparse
import os
import queue
import threading
import time
from collections import deque
from typing import Callable, Deque, List, Optional

from src.contracts import (
    DEVICE_BAUD, Clock, HapticCmd, ImuEvent, ImuSample, MotionState, RealClock,
    format_haptic_line, parse_device_line,
)

ARDUINO_VIDS = (0x2341, 0x8087, 0x2A03)  # Arduino SA, Intel (Arduino 101), Arduino.org


def find_port(preferred: Optional[str] = None) -> Optional[str]:
    """The port to use: $ARDUINO_PORT / `preferred` if it exists, else the first Arduino-looking port."""
    import serial.tools.list_ports as lp

    ports = list(lp.comports())
    want = preferred or os.environ.get("ARDUINO_PORT")
    if want and want in [p.device for p in ports]:
        return want
    for p in ports:
        text = f"{p.description} {p.manufacturer or ''}".lower()
        if "arduino" in text or p.vid in ARDUINO_VIDS:
            return p.device
    return None


class ArduinoImuLink:
    LIVE_LINES_PER_HALF_SECOND = 24   # live stream is ~16 lines per 0.5 s; a backlog delivers hundreds
    STALE_AFTER_S = 3.0               # no bytes for this long -> assume the link is dead, reconnect
    KEEPALIVE_S = 1.0                 # the board only streams while it keeps hearing from us (see the firmware)

    def __init__(
        self,
        clock: Clock,
        port: Optional[str] = None,
        baud: int = DEVICE_BAUD,
        serial_factory: Optional[Callable[[], object]] = None,   # tests inject a fake port
        drain: bool = True,
        drain_max_s: float = 60.0,
        sample_buffer: int = 1000,
        reconnect_delay_s: float = 1.0,
    ) -> None:
        self.clock = clock
        self._port, self._baud = port, baud
        self._serial_factory = serial_factory
        self._drain, self._drain_max_s = drain, drain_max_s
        self._reconnect_delay_s = reconnect_delay_s

        self._lock = threading.Lock()
        self._events: List[ImuEvent] = []
        self._samples: Deque[ImuSample] = deque(maxlen=sample_buffer)
        self._motion = MotionState.STILL
        self._tx: "queue.Queue[str]" = queue.Queue(maxsize=16)
        self._stop = threading.Event()
        self._connected = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_error: str = ""
        self.lines_seen = 0
        self.port_name: str = ""                 # the port actually opened (for the HUD / preflight)

    # ---- ImuLink contract -------------------------------------------------------------
    @property
    def motion(self) -> MotionState:
        with self._lock:
            return self._motion

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ArduinoImuLink", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._connected.clear()

    def poll(self) -> List[ImuEvent]:
        with self._lock:
            out, self._events = self._events, []
        return out

    def send(self, cmd: HapticCmd) -> None:
        """Non-blocking. Dropped while disconnected: a haptic cue that arrives late is worse than none."""
        if not self._connected.is_set():
            return
        try:
            self._tx.put_nowait(format_haptic_line(cmd))
        except queue.Full:
            pass

    def recent_samples(self, n: int = 200) -> List[ImuSample]:
        with self._lock:
            return list(self._samples)[-n:]

    # ---- extras (not in the contract) ---------------------------------------------------
    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def wait_connected(self, timeout: float = 10.0) -> bool:
        return self._connected.wait(timeout)

    # ---- internals ----------------------------------------------------------------------
    def _open(self):
        if self._serial_factory is not None:
            return self._serial_factory()
        import serial

        port = find_port(self._port)
        if port is None:
            raise OSError("no Arduino serial port found")
        self.port_name = port
        # short read timeout: the same thread also writes queued haptic lines, so this bounds their latency
        return serial.Serial(port, self._baud, timeout=0.02, write_timeout=0.5)

    def _readline(self, ser) -> str:
        return ser.readline().decode("ascii", errors="ignore")

    def _drain_backlog(self, ser) -> None:
        t0 = time.time()
        while time.time() - t0 < self._drain_max_s and not self._stop.is_set():
            window_start, lines = time.time(), 0
            while time.time() - window_start < 0.5 and not self._stop.is_set():
                if self._readline(ser).strip():
                    lines += 1
            if lines <= self.LIVE_LINES_PER_HALF_SECOND:
                return

    def _set_motion(self, state: MotionState) -> None:
        with self._lock:
            if state != self._motion:
                self._motion = state
                self._events.append(ImuEvent(self.clock.now(), motion=state))

    def _handle_line(self, line: str) -> None:
        rec = parse_device_line(line, self.clock.now())
        if rec is None:
            return
        if isinstance(rec, ImuSample):
            with self._lock:
                self._samples.append(rec)
        elif rec.motion is not None:
            self._set_motion(rec.motion)
        else:
            with self._lock:
                self._events.append(rec)

    def _run(self) -> None:
        while not self._stop.is_set():
            ser = None
            try:
                ser = self._open()
                try:
                    while self._tx.get_nowait():  # anything queued before the (re)connect is stale
                        pass
                except queue.Empty:
                    pass
                if self._drain:
                    self._drain_backlog(ser)
                ser.write(b"PING\n")  # the board answers READY plus its current motion state
                self.last_error = ""
                self._connected.set()
                self._serve(ser)
            except Exception as e:  # serial errors, unplug, no port yet: all handled by reconnecting
                self.last_error = f"{type(e).__name__}: {e}"
            finally:
                self._connected.clear()
                self._set_motion(MotionState.STILL)  # unknown now: the safe assumption
                if ser is not None:
                    try:
                        ser.close()
                    except Exception:
                        pass
            self._stop.wait(self._reconnect_delay_s)

    def _serve(self, ser) -> None:
        last_data = last_ping = time.time()
        while not self._stop.is_set():
            if time.time() - last_ping >= self.KEEPALIVE_S:
                ser.write(b"PING\n")  # keepalive: the board answers READY + its motion state
                last_ping = time.time()
            try:
                ser.write((self._tx.get_nowait()).encode("ascii"))
            except queue.Empty:
                pass
            line = self._readline(ser)
            if line:
                last_data = time.time()
                self.lines_seen += 1
                self._handle_line(line)
            elif time.time() - last_data > self.STALE_AFTER_S:
                raise TimeoutError(f"no data from the board for {self.STALE_AFTER_S:.0f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description="Live monitor for the pen probe.")
    ap.add_argument("--port")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this long (default: run until Ctrl+C)")
    ap.add_argument("--haptic", choices=[c.value for c in HapticCmd], help="send this haptic command once after connecting")
    args = ap.parse_args()

    clock = RealClock()
    link = ArduinoImuLink(clock, port=args.port)
    link.start()
    print("connecting (draining any buffered data can take a while)...")
    t0 = clock.now()
    last_status, last_n = t0, 0
    sent = False
    try:
        while args.seconds <= 0 or clock.now() - t0 < args.seconds:
            for ev in link.poll():
                print(f"  event: {'motion ' + ev.motion.value if ev.motion else 'tap ' + str(int(ev.tap))}")
            if args.haptic and link.connected and not sent:
                link.send(HapticCmd(args.haptic))
                sent = True
                print(f"  sent H,{args.haptic}")
            now = clock.now()
            if now - last_status >= 1.0:
                s = link.recent_samples(1)
                total = link.lines_seen
                latest = f"a=({s[0].ax:+.2f},{s[0].ay:+.2f},{s[0].az:+.2f})" if s else "no samples yet"
                err = f"  [{link.last_error}]" if link.last_error else ""
                print(f"{'connected' if link.connected else 'DISCONNECTED'}  motion={link.motion.value}  "
                      f"{(total - last_n) / (now - last_status):.0f} lines/s  {latest}{err}")
                last_status, last_n = now, total
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        link.stop()


if __name__ == "__main__":
    main()
