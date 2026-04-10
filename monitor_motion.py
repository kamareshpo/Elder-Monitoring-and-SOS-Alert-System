# monitor_motion.py  — Thread-safe no-movement detector
import cv2
import time
import numpy as np
from datetime import datetime

from sos_alert import send_sos
from buzzer import trigger_buzzer

NO_MOVEMENT_THRESHOLD_SECONDS = 3600   # 1 hour — adjust as needed
MIN_CONTOUR_AREA               = 1500  # ignore tiny noise blobs


def detect_motion(get_frame, cam_id: int = 0):
    """
    Compares consecutive greyscale frames.  If no significant motion is
    detected for NO_MOVEMENT_THRESHOLD_SECONDS, an SOS alert is fired.
    """
    print(f"👁  Motion Detection running — Camera {cam_id}")

    last_movement  = datetime.now()
    prev_gray      = None
    alert_sent     = False

    while True:
        frame = get_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0
        )

        if prev_gray is None:
            prev_gray = gray
            time.sleep(0.3)
            continue

        diff      = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh    = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        motion = any(cv2.contourArea(c) > MIN_CONTOUR_AREA for c in contours)

        if motion:
            last_movement = datetime.now()
            alert_sent    = False

        elapsed = (datetime.now() - last_movement).total_seconds()
        if elapsed > NO_MOVEMENT_THRESHOLD_SECONDS and not alert_sent:
            hours = int(elapsed // 3600)
            photo_path = f"no_motion_cam{cam_id}.jpg"
            cv2.imwrite(photo_path, frame)
            send_sos(
                f"⚠️  No movement detected for {hours} hour(s)!",
                photo_path=photo_path,
                cam_id=cam_id,
            )
            trigger_buzzer()
            alert_sent = True

        prev_gray = gray
        time.sleep(0.3)
