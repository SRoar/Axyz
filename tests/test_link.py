"""ArduinoImuLink against a fake serial port: no hardware needed."""
import threading
import time
from collections import deque

import pytest

from src.arduino_link import ArduinoImuLink
from src.contracts import HapticCmd, MotionState, RealClock, Tap, parse_device_line, ImuSample, ImuEvent


class FakePort:
    """Just enough of pyserial.Serial: readline() / write() / close(), fed from a deque of lines."""

    def __init__(self, lines=(), fail_when_empty=False):
        self.rx = deque(l if isinstance(l, bytes) else (l + "\n").encode() for l in lines)
        self.tx = []
        self.closed = False
        self.fail_when_empty = fail_when_empty
        self._lock = threading.Lock()

    def feed(self, *lines):
        with self._lock:
            for l in lines:
                self.rx.append((l + "\n").encode())

    def readline(self):
        if self.closed:
            raise OSError("closed")
        with self._lock:
            if self.rx:
                return self.rx.popleft()
        if self.fail_when_empty:
            raise OSError("device disconnected")
        time.sleep(0.005)
        return b""

    def write(self, data):
        if self.closed:
            raise OSError("closed")
        self.tx.append(data)
        return len(data)

    def close(self):
        self.closed = True


def wait_for(cond, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.01)
    return False


def make_link(ports, **kw):
    """A link whose serial_factory hands out `ports` in order (then keeps failing like an unplugged board)."""
    it = iter(ports)

    def factory():
        try:
            return next(it)
        except StopIteration:
            raise OSError("no more ports")

    kw.setdefault("drain", False)
    kw.setdefault("reconnect_delay_s", 0.05)
    return ArduinoImuLink(RealClock(), serial_factory=factory, **kw)


def test_parses_samples_and_events_in_order():
    port = FakePort(["READY", "S,100,0.1,0.2,0.9", "M,MOVING", "S,130,0.1,0.3,0.9", "T,2", "garbage,,,", "M,NOPE"])
    link = make_link([port])
    link.start()
    try:
        assert wait_for(lambda: len(link.recent_samples()) == 2 and link.motion == MotionState.MOVING)
        assert wait_for(lambda: len(link._events) >= 2)
        ev = link.poll()
        assert [(e.motion, e.tap) for e in ev] == [(MotionState.MOVING, Tap.NONE), (None, Tap.DOUBLE)]
        assert link.poll() == []  # events are delivered once
        s = link.recent_samples()
        assert (s[0].ax, s[0].ay, s[0].az) == (0.1, 0.2, 0.9)
    finally:
        link.stop()


def test_repeated_motion_lines_do_not_repeat_events():
    port = FakePort(["M,WRITING", "M,WRITING", "M,STILL", "M,STILL"])
    link = make_link([port])
    link.start()
    try:
        assert wait_for(lambda: link.motion == MotionState.STILL and len(link._events) == 2)
        assert [e.motion for e in link.poll()] == [MotionState.WRITING, MotionState.STILL]
    finally:
        link.stop()


def test_ping_on_connect_and_haptic_lines():
    port = FakePort()
    link = make_link([port])
    assert not link.connected
    link.send(HapticCmd.LOCK)  # before connecting: dropped, never queued
    link.start()
    try:
        assert link.wait_connected(2.0)
        assert wait_for(lambda: b"PING\n" in port.tx)
        link.send(HapticCmd.GUIDE_LEFT)
        link.send(HapticCmd.OFF)
        assert wait_for(lambda: b"H,GUIDE_LEFT\n" in port.tx and b"H,OFF\n" in port.tx)
        assert b"H,LOCK\n" not in port.tx
        assert port.tx.index(b"H,GUIDE_LEFT\n") < port.tx.index(b"H,OFF\n")
    finally:
        link.stop()


def test_reconnects_after_the_board_drops_off_usb():
    first = FakePort(["M,WRITING"], fail_when_empty=True)  # then "unplugs"
    second = FakePort(["S,5,0,0,1,170", "M,MOVING"])
    link = make_link([first, second])
    link.start()
    try:
        assert wait_for(lambda: second.tx and len(link.recent_samples()) == 1 and link.motion == MotionState.MOVING, 4.0)
        assert first.closed
        motions = [e.motion for e in link.poll()]
        # WRITING, then STILL on disconnect (state unknown, safe default), then MOVING from the new port
        assert motions == [MotionState.WRITING, MotionState.STILL, MotionState.MOVING]
    finally:
        link.stop()


def test_stale_haptics_are_not_replayed_after_reconnect():
    first = FakePort(fail_when_empty=True)
    second = FakePort()
    link = make_link([first, second])
    link.start()
    try:
        assert wait_for(lambda: second.tx, 4.0)  # connected to the second port (PING sent)
        assert all(not t.startswith(b"H,") for t in second.tx)
    finally:
        link.stop()


def test_silent_board_is_treated_as_a_dead_link():
    quiet = FakePort()
    second = FakePort(["S,1,0,0,1,170"])
    link = make_link([quiet, second])
    link.STALE_AFTER_S = 0.2
    link.start()
    try:
        assert wait_for(lambda: len(link.recent_samples()) == 1, 4.0)
        assert quiet.closed
    finally:
        link.stop()


def test_backlog_is_drained_before_going_live():
    stale = ["S,%d,0,0,1,170" % i for i in range(2500)] + ["M,WRITING", "T,3"]
    port = FakePort(stale)
    link = make_link([port], drain=True)
    link.start()
    try:
        assert link.wait_connected(5.0)
        port.feed("S,999999,0.1,0.1,0.9,170", "M,MOVING")  # live data after the drain
        assert wait_for(lambda: link.motion == MotionState.MOVING)
        ev = link.poll()
        assert [e.motion for e in ev if e.motion] == [MotionState.MOVING]  # stale WRITING ignored
        assert all(e.tap == Tap.NONE for e in ev)                           # stale triple tap ignored
        assert all(s.t >= 0 for s in link.recent_samples())
        assert len(link.recent_samples()) == 1                              # only the live sample was kept
    finally:
        link.stop()


def test_recent_samples_is_bounded_and_newest_last():
    port = FakePort(["S,%d,0,0,1,170" % i for i in range(50)])
    link = make_link([port], sample_buffer=20)
    link.start()
    try:
        assert wait_for(lambda: link.lines_seen >= 50)
        s = link.recent_samples(5)
        assert len(s) == 5
        assert len(link.recent_samples(1000)) == 20
        assert s[-1].t >= s[0].t
    finally:
        link.stop()


def test_stop_is_prompt_and_idempotent():
    link = make_link([FakePort()])
    link.start()
    assert link.wait_connected(2.0)
    t0 = time.time()
    link.stop()
    link.stop()
    assert time.time() - t0 < 2.0
    assert not link.connected


def test_no_board_attached_never_raises_and_stays_still():
    link = make_link([])  # every open attempt fails
    link.start()
    try:
        time.sleep(0.3)
        assert not link.connected
        assert link.poll() == []
        assert link.motion == MotionState.STILL
        assert link.last_error  # says why
        link.send(HapticCmd.WARN)  # must not raise or block
    finally:
        link.stop()


def test_contract_parser_agrees_with_what_the_firmware_prints():
    # the exact shapes illumin_pen.ino emits
    s = parse_device_line("S,254998,-0.281,-1.031,-0.140", 1.0)
    assert isinstance(s, ImuSample) and (s.ax, s.ay, s.az) == (-0.281, -1.031, -0.14)
    old = parse_device_line("S,254998,-0.281,-1.031,-0.140,170", 1.0)   # an older sketch that still sent light
    assert isinstance(old, ImuSample) and (old.ax, old.ay, old.az) == (-0.281, -1.031, -0.14)
    assert parse_device_line("M,WRITING", 1.0).motion == MotionState.WRITING
    assert parse_device_line("T,1", 1.0).tap == Tap.SINGLE
    assert parse_device_line("T,2", 1.0).tap == Tap.DOUBLE
    for junk in ("READY", "ERR,accel_lost", "", "S,1,2", "T,x"):
        assert parse_device_line(junk, 1.0) is None


def test_keepalive_pings_keep_the_board_streaming():
    """The firmware only streams while it keeps hearing from the PC, so the link must ping regularly."""
    port = FakePort()
    link = make_link([port])
    link.KEEPALIVE_S = 0.1
    link.start()
    try:
        assert link.wait_connected(2.0)
        assert wait_for(lambda: port.tx.count(b"PING\n") >= 4, 3.0)
    finally:
        link.stop()
