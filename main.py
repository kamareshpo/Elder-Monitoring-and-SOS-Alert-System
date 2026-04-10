# main.py  — Elder Monitoring System  |  3-Camera Grid with Night Vision
from threading import Thread
import cv2
import numpy as np
import time

from camera_manager import CameraManager
from fall_detection  import detect_fall
from gesture_sos     import detect_gesture
from monitor_motion  import detect_motion
from night_vision    import enhance_night_vision
from sos_alert       import send_sos

send_sos("TEST ALERT FROM SYSTEM — startup")

# ── Configuration ─────────────────────────────────────────────────────────────
CAMERA_IDS   = [0, 1, 2]    # change to [0] if you only have one camera

GRID_CELL_W  = 640
GRID_CELL_H  = 480
DISPLAY_W    = 1280
DISPLAY_H    = 760

# ── Initialise camera manager ─────────────────────────────────────────────────
cam_mgr = CameraManager(camera_ids=CAMERA_IDS)


# ── Detection worker threads ──────────────────────────────────────────────────
#
#  Both detectors now receive a set_annotated_frame callback so their
#  skeleton / hand-landmark overlays are routed back into the display grid.

def _run_fall(cam_id):
    """Fall detector — restarts automatically if it ever crashes."""
    while cam_mgr.running:
        try:
            detect_fall(
                get_frame=lambda cid=cam_id: cam_mgr.get_frame(cid),
                cam_id=cam_id,
                set_annotated_frame=cam_mgr.set_annotated_frame,
            )
        except Exception as e:
            print(f"[FALL cam{cam_id}] crashed: {e} — restarting in 2 s")
            time.sleep(2)

def _run_gesture(cam_id):
    """Gesture detector."""
    detect_gesture(
        get_frame=lambda cid=cam_id: cam_mgr.get_frame(cid),
        cam_id=cam_id,
        set_annotated_frame=cam_mgr.set_annotated_frame,
    )

def _run_motion(cam_id):
    """Motion / no-movement detector."""
    detect_motion(
        lambda cid=cam_id: cam_mgr.get_frame(cid),
        cam_id=cam_id,
    )


# ── 3-Camera grid live view ───────────────────────────────────────────────────

def _label_frame(frame, cam_id: int, night: bool) -> np.ndarray:
    """Resize and stamp the camera tile header."""
    tile   = cv2.resize(frame, (GRID_CELL_W, GRID_CELL_H))
    mode   = "NIGHT" if night else "LIVE"
    colour = (0, 255, 0) if night else (0, 255, 255)
    cv2.rectangle(tile, (0, 0), (GRID_CELL_W, 38), (0, 0, 0), -1)
    cv2.putText(tile, f" CAM {cam_id}  [{mode}]", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
    return tile


def run_live_grid():
    """
    Display loop (runs on the main thread).
    Keys:  n — toggle night vision   |   q — quit
    """
    night_mode = False

    while cam_mgr.running:
        tiles = []
        for cam_id in CAMERA_IDS:
            # Use annotated frame (with skeleton overlay) when available
            frame = cam_mgr.get_display_frame(cam_id)
            if frame is None:
                frame = cam_mgr.offline_placeholder(cam_id)
            if night_mode:
                frame = enhance_night_vision(frame)
            tiles.append(_label_frame(frame, cam_id, night_mode))

        # Layout: tiles[0] + tiles[1] top row, tiles[2] centred bottom row
        top_row = np.hstack([tiles[0], tiles[1]])
        pad     = (GRID_CELL_W * 2 - GRID_CELL_W) // 2      # 320 px each side
        bottom_row = np.hstack([
            np.zeros((GRID_CELL_H, pad, 3), dtype=np.uint8),
            tiles[2],
            np.zeros((GRID_CELL_H, pad, 3), dtype=np.uint8),
        ])
        grid = np.vstack([top_row, bottom_row])

        # Scale + header bar
        display = cv2.resize(grid, (DISPLAY_W, DISPLAY_H - 40))
        header  = np.zeros((40, DISPLAY_W, 3), dtype=np.uint8)
        cv2.putText(
            header,
            "ELDER MONITORING SYSTEM  |  [n] Night Vision  [q] Quit",
            (30, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1,
        )
        full_display = np.vstack([header, display])
        cv2.imshow("Elder Monitoring — 3-Camera Feed", full_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Shutting down...")
            cam_mgr.stop()
            break
        elif key == ord("n"):
            night_mode = not night_mode
            print(f"Night Vision: {'ON' if night_mode else 'OFF'}")

    cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Elder Monitoring System starting...")
    cam_mgr.start()

    for cid in CAMERA_IDS:
        Thread(target=_run_fall,    args=(cid,), daemon=True).start()
        Thread(target=_run_gesture, args=(cid,), daemon=True).start()
        Thread(target=_run_motion,  args=(cid,), daemon=True).start()

    # Live grid runs on main thread (OpenCV windows require the main thread)
    run_live_grid()
    print("System stopped.")