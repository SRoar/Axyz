"""Step 1: find the Logitech, lock its settings, and get the full frame rate into the laptop.

Usage:
    python tools/camera_check.py --list            # camera NAMES + indexes (the tracker picks the Logitech by name)
    python tools/camera_check.py                   # live preview using data/camera.json
    python tools/camera_check.py --bench           # measure DELIVERED FPS per format / resolution / exposure
    python tools/camera_check.py --bench --save    # ...and save the best settings to data/camera.json
    python tools/camera_check.py --index 1 --width 1280 --height 720 --fps 30
    python tools/camera_check.py --scan            # show one frame from each index

Preview keys:
    [ / ]  manual exposure darker / brighter       a = back to auto exposure
    - / =  gain down / up (brightens a short exposure without costing FPS)
    f      toggle autofocus                         p = driver settings dialog (Windows/DirectShow)
    w      write the current settings to data/camera.json
    s      save frame to data/camera_check.png      ESC = quit

Notes:
- Every Dev 2 tool and the PenTracker read data/camera.json, so tune here once.
- Requested settings are often ignored silently; the preview prints what actually took effect,
  and the FPS shown is frames actually delivered (not the driver's claim).
- Low FPS checklist: MJPG not taking effect (USB bandwidth), auto-exposure in a dim room (each
  frame exposed > 33 ms), a virtual camera (NVIDIA Broadcast / OBS) in the path, or another app
  holding the camera. --bench measures each of these.
- On Windows close NVIDIA Broadcast, Teams, Zoom and the Camera app first: only one program can
  hold the camera.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.webcam import (  # noqa: E402
    CAMERA_CFG_PATH, CameraError, Webcam, add_camera_args, backend_id, config_from_args,
    is_virtual, list_devices, resolve_device, save_camera_config,
)

BENCH_SIZES = [(1280, 720), (1920, 1080), (640, 480)]     # preference order
BENCH_EXPOSURES = [-4, -5, -6, -7, "auto"]                  # DirectShow log2(seconds); auto last restores it
BENCH_GAINS = [64, 128, 192, 255]
BRIGHTNESS_OK, BRIGHTNESS_TARGET = 90, 125


def list_cameras(cfg):
    apis = [cv2.CAP_DSHOW, cv2.CAP_MSMF] if sys.platform == "win32" else [backend_id()]
    for api in apis:
        devs = list_devices(api)
        name = {cv2.CAP_DSHOW: "DirectShow", cv2.CAP_MSMF: "Media Foundation"}.get(api, str(api))
        print(f"{name}:")
        if not devs:
            print("  (names unavailable: pip install cv2_enumerate_cameras)")
        for idx, n in devs:
            tag = "   <- virtual camera: avoid (latency, capped FPS)" if is_virtual(n) else ""
            print(f"  index {idx}: {n}{tag}")
    idx, n = resolve_device(cfg, backend_id(cfg.backend))
    print(f"\nwith data/camera.json (name={cfg.name!r}, index={cfg.index}) the tracker opens: {n} (index {idx})")


def scan(cfg, max_index=5):
    print("Scanning camera indexes (a window shows each one for ~1.5 s)...")
    names = dict(list_devices(backend_id(cfg.backend)))
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend_id(cfg.backend))
        if not cap.isOpened():
            print(f"  index {i}: not available ({names.get(i, '?')})")
            continue
        ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            print(f"  index {i}: OK  {w}x{h}  {names.get(i, '')}")
            cv2.putText(frame, f"index {i} {names.get(i, '')}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.imshow("scan", frame)
            cv2.waitKey(1500)
        else:
            print(f"  index {i}: opened but no frame ({names.get(i, '?')}: busy in another app?)")
        cap.release()
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------- bench
def measure(cfg, seconds=2.5, warmup=1.0):
    cam = Webcam(cfg, verbose=False)
    try:
        cam.start(first_frame_timeout=6.0)
    except CameraError as e:
        return {"error": str(e).splitlines()[0]}
    try:
        time.sleep(warmup)
        f0, t0 = cam.latest(), time.perf_counter()
        time.sleep(seconds)
        f1, t1 = cam.latest(), time.perf_counter()
        img = f1.image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return {
            "fps": (f1.id - f0.id) / (t1 - t0),
            "size": (img.shape[1], img.shape[0]),
            "fourcc": cam.info.get("fourcc"),
            "brightness": float(gray.mean()),
            "exposure_readback": cam.info.get("exposure"),
            "device": cam.info.get("device"),
        }
    finally:
        cam.stop()


def _row(label, r):
    if "error" in r:
        return f"  {label:<34} ERROR {r['error']}"
    return (f"  {label:<34} {r['fps']:5.1f} fps  got {r['size'][0]}x{r['size'][1]} {r['fourcc']:<4}  "
            f"brightness {r['brightness']:5.1f}")


def bench(cfg, save):
    target = cfg.fps
    dshow = backend_id(cfg.backend) in (cv2.CAP_DSHOW, cv2.CAP_MSMF)
    print(f"Benchmarking {resolve_device(cfg, backend_id(cfg.backend))[1]} (target {target} fps). "
          "Keep the page + pen in view, demo lighting on.\n")

    print("1) format x resolution (short manual exposure, so only USB bandwidth / decoding limit FPS):")
    fmt_results = []
    for (w, h) in BENCH_SIZES:
        for four in ("MJPG", ""):
            c = cfg.replace(width=w, height=h, fourcc=four, exposure=-7 if dshow else "auto")
            r = measure(c)
            fmt_results.append((c, r))
            print(_row(f"{w}x{h} {four or 'driver default'}", r), flush=True)
    ok = [(c, r) for c, r in fmt_results if "error" not in r]
    if not ok:
        print("\nNo configuration delivered frames. Run --list and close other camera apps.")
        return
    good = [(c, r) for c, r in ok if r["fps"] >= 0.9 * target and r["size"] == (c.width, c.height)]
    best_c, best_r = (good[0] if good else max(ok, key=lambda cr: cr[1]["fps"]))
    print(f"   -> {best_c.width}x{best_c.height} {best_c.fourcc or 'default'} ({best_r['fps']:.1f} fps)")

    print("\n2) exposure (auto-exposure in a dim room is the usual reason for 15 fps):")
    exp_results = []
    for e in (BENCH_EXPOSURES if dshow else ["auto"]):
        c = best_c.replace(exposure=e)
        r = measure(c)
        exp_results.append((c, r))
        print(_row(f"exposure {e}", r), flush=True)
    exp_ok = [(c, r) for c, r in exp_results if "error" not in r]
    auto = next(((c, r) for c, r in exp_ok if c.exposure == "auto"), None)
    fast = [(c, r) for c, r in exp_ok if r["fps"] >= 0.9 * target]
    if auto and auto[1]["fps"] >= 0.9 * target:
        chosen = auto
        note = "auto exposure keeps up; re-run --bench at the venue (dimmer light can halve the FPS)"
    elif fast:
        chosen = max(fast, key=lambda cr: cr[1]["brightness"])
        note = "auto exposure throttles the frame rate here, so exposure is locked"
    else:
        chosen = max(exp_ok, key=lambda cr: cr[1]["fps"]) if exp_ok else (best_c, best_r)
        note = "nothing reaches the target; add light or lower the resolution"
    c, r = chosen
    if c.exposure != "auto" and r.get("brightness", 255) < BRIGHTNESS_OK:
        print(f"\n3) gain (locked exposure {c.exposure} is dark, brightness {r['brightness']:.0f}):")
        gain_results = []
        for gain in BENCH_GAINS:
            gc = c.replace(gain=gain)
            gr = measure(gc)
            print(_row(f"exposure {c.exposure} gain {gain}", gr), flush=True)
            if "error" not in gr and gr["fps"] >= 0.9 * target:
                gain_results.append((gc, gr))
        if gain_results:
            c, r = min(gain_results, key=lambda cr: abs(cr[1]["brightness"] - BRIGHTNESS_TARGET))
    print(f"\nRECOMMENDED: {c.width}x{c.height} {c.fourcc or 'default'} exposure={c.exposure} gain={c.gain} "
          f"-> {r.get('fps', 0):.1f} fps   ({note})")
    if r.get("brightness", 255) < 60:
        print("   image is dark (brightness < 60): add light, or press p in the preview for the driver dialog")
    if save:
        save_camera_config(c, CAMERA_CFG_PATH)
        print(f"saved {CAMERA_CFG_PATH}")
    else:
        print("re-run with --save to write data/camera.json")


# --------------------------------------------------------------------------- preview
def preview(cfg):
    cam = Webcam(cfg)
    try:
        cam.start()
    except CameraError as e:
        print(e)
        return
    print("--- property read-back (0 or -1 can mean 'unsupported') ---")
    for k, v in cam.info.items():
        print(f"  {k:12s} {v}")
    exposure = cfg.exposure if isinstance(cfg.exposure, (int, float)) else None
    autofocus = cfg.autofocus
    gain = cfg.gain
    shown, t_show, disp_fps, last_print, last_id = 0, time.perf_counter(), 0.0, 0.0, 0
    try:
        while True:
            f = cam.wait(last_id, timeout=1.0)
            if f is None:
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue
            last_id = f.id
            shown += 1
            now = time.perf_counter()
            if now - t_show >= 1.0:
                disp_fps, shown, t_show = shown / (now - t_show), 0, now
            cam_fps = cam.fps
            if now - last_print >= 1.0:
                last_print = now
                print(f"delivered {cam_fps:5.1f} fps | displayed {disp_fps:5.1f} fps | exposure {exposure or 'auto'}")
            frame = f.image.copy()
            h, w = frame.shape[:2]
            col = (0, 255, 0) if cam_fps >= 25 else (0, 0, 255)
            cv2.putText(frame, f"{w}x{h} {cam.info.get('fourcc')}  camera {cam_fps:.1f} fps  "
                        f"exposure {exposure if exposure is not None else 'auto'}",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
            cv2.putText(frame, f"brightness {cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY).mean():.0f}  "
                        "[ ] exposure  a auto  f focus  p driver  w save  s frame",
                        (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            # crosshair: check the camera is square above the page
            cv2.line(frame, (w // 2, 0), (w // 2, h), (0, 255, 255), 1)
            cv2.line(frame, (0, h // 2), (w, h // 2), (0, 255, 255), 1)
            cv2.imshow("camera_check", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key in (ord("["), ord("]")):
                exposure = (-6 if exposure is None else exposure) + (-1 if key == ord("[") else 1)
                cam.set_exposure(exposure)
            elif key == ord("a"):
                exposure = None
                cam.set_exposure("auto")
            elif key in (ord("-"), ord("=")):
                gain = min(255.0, max(0.0, (gain if gain is not None else 64.0) + (-16 if key == ord("-") else 16)))
                cam.set_property(cv2.CAP_PROP_GAIN, gain)
                print(f"gain {gain:.0f}")
            elif key == ord("f"):
                autofocus = not bool(autofocus)
                cam.set_property(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
                print(f"autofocus {'on' if autofocus else 'off'}")
            elif key == ord("p"):
                cam.open_driver_settings()
            elif key == ord("w"):
                out = cfg.replace(width=w, height=h, exposure=exposure if exposure is not None else "auto",
                                  gain=gain, autofocus=autofocus, index=cam.info.get("index", cfg.index))
                save_camera_config(out, CAMERA_CFG_PATH)
                print(f"saved {CAMERA_CFG_PATH}: {out}")
            elif key == ord("s"):
                os.makedirs("data", exist_ok=True)
                cv2.imwrite("data/camera_check.png", f.image)
                print("saved data/camera_check.png")
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    add_camera_args(ap)
    ap.add_argument("--list", action="store_true", help="list camera names and indexes")
    ap.add_argument("--scan", action="store_true", help="show one frame from each index")
    ap.add_argument("--bench", action="store_true", help="measure delivered FPS per setting")
    ap.add_argument("--save", action="store_true", help="with --bench: write the best settings")
    args = ap.parse_args()
    cfg = config_from_args(args)
    if args.list:
        list_cameras(cfg)
    elif args.scan:
        scan(cfg)
    elif args.bench:
        bench(cfg, args.save)
    else:
        preview(cfg)
