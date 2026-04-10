# camera_manager.py — adds annotated-frame channel so detector overlays appear in the grid
import cv2
import threading
import time
import numpy as np


class CameraManager:
    """Manages up to 3 simultaneous camera streams with thread-safe frame access.

    Two frame channels per camera:
      • raw frames  — written by capture loop, used by detectors
      • annotated frames — written by detectors, used by the display grid
    """

    def __init__(self, camera_ids=[0, 1, 2]):
        self.camera_ids = camera_ids
        self.running    = True

        # Raw frames (from hardware)
        self.frames       = {c: None for c in camera_ids}
        self.locks        = {c: threading.Lock() for c in camera_ids}
        self.connected    = {c: False for c in camera_ids}

        # Annotated frames (written back by fall / gesture detectors)
        self.annotated_frames = {c: None for c in camera_ids}
        self.anno_locks       = {c: threading.Lock() for c in camera_ids}

    # ── Start capture threads ────────────────────────────────────

    def start(self):
        for cam_id in self.camera_ids:
            t = threading.Thread(target=self._capture_loop,
                                 args=(cam_id,), daemon=True)
            t.start()
        time.sleep(1.5)

    def _capture_loop(self, cam_id):
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            print(f"[CAM {cam_id}] Not connected — showing OFFLINE placeholder.")
            self.connected[cam_id] = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,          30)
        self.connected[cam_id] = True
        print(f"[CAM {cam_id}] Connected.")

        while self.running:
            ret, frame = cap.read()
            if ret:
                with self.locks[cam_id]:
                    self.frames[cam_id] = frame
            else:
                print(f"[CAM {cam_id}] Frame grab failed.")
                self.connected[cam_id] = False
                break
            time.sleep(0.01)
        cap.release()

    # ── Raw frame access ─────────────────────────────────────────

    def get_frame(self, cam_id):
        """Return the latest raw frame (used by detectors)."""
        with self.locks[cam_id]:
            f = self.frames[cam_id]
            return f.copy() if f is not None else None

    # ── Annotated frame access ───────────────────────────────────

    def set_annotated_frame(self, cam_id, frame):
        """Detectors call this to write their overlay back for display."""
        with self.anno_locks[cam_id]:
            self.annotated_frames[cam_id] = frame.copy()

    def get_display_frame(self, cam_id):
        """
        Returns the annotated frame if one exists, otherwise the raw frame.
        Used by the display grid in main.py.
        """
        with self.anno_locks[cam_id]:
            af = self.annotated_frames[cam_id]
        if af is not None:
            return af.copy()
        return self.get_frame(cam_id)

    # ── Offline placeholder ──────────────────────────────────────

    def offline_placeholder(self, cam_id):
        ph = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(ph, f"CAM {cam_id}  OFFLINE", (120, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 200), 2)
        cv2.putText(ph, "Check connection", (180, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
        return ph

    def stop(self):
        self.running = False