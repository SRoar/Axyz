"""Central orchestrator with live HUD, Gemini grounding, and spoken navigation cues."""

import time
import cv2
import numpy as np
from src.config import config
from src.camera import VisionTracker
from src.vision_agent import VisionAgent, TargetBoundingBox
from src.mock_bridge import MockHapticBridge, Telemetry
from src.voice import VoiceEngine


def describe_direction(dx_cm: float, dy_cm: float, dist_cm: float) -> str:
    """
    Restricts directions strictly to: Up, Down, Left, Right.
    Speaks the dominant (farthest) direction first.
    
    Coordinate convention (OpenCV origin at top-left):
      dy > 0: target is further down the page -> "Down"
      dy < 0: target is higher up the page   -> "Up"
      dx > 0: target is to the right         -> "Right"
      dx < 0: target is to the left          -> "Left"
    """
    DEADZONE_CM = 1.0  # Tolerance threshold for axis alignment

    abs_x = abs(dx_cm)
    abs_y = abs(dy_cm)

    # If both axes are within deadzone, target is reached
    if abs_x < DEADZONE_CM and abs_y < DEADZONE_CM:
        return "You are on the target."

    # Compare which axis error is larger
    if abs_y >= abs_x:
        # Farthest direction is vertical
        direction = "Down" if dy_cm > 0 else "Up"
        dist_inches = max(1, int(round(abs_y / 2.54)))
    else:
        # Farthest direction is horizontal
        direction = "Right" if dx_cm > 0 else "Left"
        dist_inches = max(1, int(round(abs_x / 2.54)))

    unit_str = "inch" if dist_inches == 1 else "inches"
    return f"Move {dist_inches} {unit_str} {direction}."


def main():
    tracker = VisionTracker()
    tracker.calibrate_desk_plane()
    agent = VisionAgent()
    bridge = MockHapticBridge()
    voice = VoiceEngine()

    target_norm: tuple[float, float] | None = None
    target_box: TargetBoundingBox | None = None
    has_locked = False
    last_spoken_nav_time = 0.0

    print("\n--- Illumin Voice Navigation Ready ---")
    # print("Speak naturally: 'Find signature', 'Where do I sign', 'Find date'")
    print("Press keyboard shortcuts: [s] = Signature | [d] = Date | [c] = Clear | [q] = Quit\n")

    while tracker.is_opened():
        ret, frame, depth_map = tracker.read_frame()
        if not ret or frame is None:
            continue

        h, w = frame.shape[:2]

        # 1. Listen for voice commands (disabled: pen taps drive the flow now)
        query = None
        # spoken_cmd = voice.poll_command()
        # if spoken_cmd:
        #     if any(k in spoken_cmd for k in ["signature", "sign"]):
        #         query = "signature line"
        #     elif any(k in spoken_cmd for k in ["date", "day"]):
        #         query = "date field"
        #     elif "clear" in spoken_cmd:
        #         target_norm, target_box = None, None
        #         voice.speak("Target cleared.")

        # Keyboard fallback overrides
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            target_norm, target_box = None, None
            print("[Simulator] Target cleared.")
        elif key in (ord("s"), ord("d")):
            query = "signature line" if key == ord("s") else "date line"

        # 2. Localize Target with Gemini
        if query:
            voice.speak(f"Locating {query}.")
            t_norm, box = agent.locate_target(frame, query)
            if t_norm and box:
                target_norm, target_box = t_norm, box
                voice.speak(f"Target found. Place your hand on the page.")
                last_spoken_nav_time = 0.0  # Force immediate cue once hand is seen
                has_locked = False
            else:
                voice.speak("I could not find that target on the document.")

        # 3. Track Pointer / Hand
        hand_pose, elevation_cm, _ = tracker.estimate_hand_pose(frame, depth_map)
        hand_px = (int(hand_pose[0] * w), int(hand_pose[1] * h)) if hand_pose else None
        target_px = (int(target_norm[0] * w), int(target_norm[1] * h)) if target_norm else None
        telemetry = None

        # 4. Spoken Direction Guidance Loop
        if hand_pose and target_norm:
            telemetry = bridge.compute_guidance(hand_pose, target_norm, elevation_cm)

            if telemetry.is_locked:
                if not has_locked:
                    voice.speak("Target reached. You are directly on the line.")
                    has_locked = True
            else:
                has_locked = False
                now = time.time()
                # Provide spoken updates every 3.5 seconds so speech doesn't overlap
                if now - last_spoken_nav_time > 3.5:
                    cue = describe_direction(telemetry.dx_cm, telemetry.dy_cm, telemetry.dist_cm)
                    voice.speak(cue)
                    last_spoken_nav_time = now

        # 5. Draw HUD Visuals
        draw_hud(frame, telemetry, target_px, hand_px, target_box)
        cv2.imshow("Illumin Workspace Simulator", frame)

    tracker.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()