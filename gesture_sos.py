# gesture_sos.py — NV-fused hand detection | boost retry | MP4 night-vision clip
# ───────────────────────────────────────────────────────────────────────────────
#  NEW: Auto-detects dark scenes; in darkness the display frame is rendered in
#  Night-Vision green/thermal so hand landmarks and the OSD remain visible.
#  MediaPipe runs on the contrast-boosted frame for reliable detection even with
#  no ambient lighting.
# ───────────────────────────────────────────────────────────────────────────────

import cv2
import mediapipe as mp
import time
import threading
import os
import numpy as np

from sos_alert import send_sos
from buzzer import trigger_buzzer
from video_recorder import record_video_with_audio
from night_vision import (
    record_night_vision_clip,
    boost_for_detection,
    enhance_night_vision,
    is_dark_scene,
)

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

TIPS = [4, 8, 12, 16, 20]
PIPS = [3, 6, 10, 14, 18]

# ── Thresholds ───────────────────────────────────────────────────────────────
WRIST_Y_RAISED     = 0.45
FINGERTIP_Y_RAISED = 0.52
PALM_HOLD_SECONDS  = 2.0

HEART_WRIST_Y_MIN  = 0.38
HEART_WRIST_Y_MAX  = 0.68
HEART_WRIST_X_MIN  = 0.15
HEART_WRIST_X_MAX  = 0.85
HEART_HOLD_SECONDS = 3.0

FIST_Y_MIN         = 0.50
FIST_Y_MAX         = 0.75
FIST_HOLD_SECONDS  = 3.0

COOLDOWN_SECONDS   = 15
NO_DETECT_RETRY    = 3
DARK_THRESHOLD     = 60      # mean luma below this = dark scene
_NV_MODE           = "green" # 'green' | 'thermal' | 'white'

# Drawing specs
_LM_SPEC   = mp_draw.DrawingSpec(color=(255, 220, 0),  thickness=2, circle_radius=4)
_CONN_SPEC = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)


class GestureSOSDetector:
    def __init__(self, cam_id: int = 0, set_annotated_frame=None):
        self.cam_id    = cam_id
        self._set_anno = set_annotated_frame

        # Normal-light hands model
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
        )
        # Low-light / boosted hands model (lower thresholds)
        self._hands_boost = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self.cooldown       = COOLDOWN_SECONDS
        self.last_sos_time  = 0

        # Register with fall_detection so it can suppress fall alerts after a gesture SOS
        try:
            import fall_detection as _fd
            _fd._gesture_detector_registry.append(self)
        except Exception:
            pass
        self._fist_start    = None
        self._palm_start    = None
        self._heart_start   = None
        self._missed_frames = 0

    # ── Finger counting ──────────────────────────────────────────────────────

    def _count_raised(self, lm, label: str) -> int:
        count = 0
        if label == "Right":
            count += 1 if lm.landmark[4].x < lm.landmark[3].x else 0
        else:
            count += 1 if lm.landmark[4].x > lm.landmark[3].x else 0
        for tip, pip in zip(TIPS[1:], PIPS[1:]):
            count += 1 if lm.landmark[tip].y < lm.landmark[pip].y else 0
        return count

    def _is_open_palm(self, lm, label): return self._count_raised(lm, label) >= 4
    def _is_fist(self,      lm, label): return self._count_raised(lm, label) <= 1

    # ── Position checks ──────────────────────────────────────────────────────

    def _arm_is_raised(self, lm) -> bool:
        if lm.landmark[0].y > WRIST_Y_RAISED:
            return False
        return sum(1 for tip in TIPS[1:]
                   if lm.landmark[tip].y < FINGERTIP_Y_RAISED) >= 3

    def _hand_on_heart(self, lm) -> bool:
        wx, wy = lm.landmark[0].x, lm.landmark[0].y
        if not (HEART_WRIST_Y_MIN <= wy <= HEART_WRIST_Y_MAX):
            return False
        if not (HEART_WRIST_X_MIN <= wx <= HEART_WRIST_X_MAX):
            return False
        return sum(1 for tip in TIPS[1:]
                   if lm.landmark[tip].y >= lm.landmark[0].y - 0.10) >= 3

    # ── NV-aware hand detection ──────────────────────────────────────────────

    def _run_hands(self, raw_frame: np.ndarray):
        """
        Returns (result, display_frame).
        • In dark scenes: display_frame is NV-enhanced; detection runs on boost.
        • Retry with boosted frame if hands are lost for NO_DETECT_RETRY frames.
        """
        dark = is_dark_scene(raw_frame, DARK_THRESHOLD)
        display_frame = (
            enhance_night_vision(raw_frame, mode=_NV_MODE)
            if dark else raw_frame.copy()
        )

        detect_src = boost_for_detection(raw_frame) if dark else raw_frame
        rgb        = cv2.cvtColor(detect_src, cv2.COLOR_BGR2RGB)
        model      = self._hands_boost if dark else self.hands
        result     = model.process(rgb)

        if result.multi_hand_landmarks:
            self._missed_frames = 0
            return result, display_frame

        self._missed_frames += 1
        if self._missed_frames >= NO_DETECT_RETRY:
            rgb2    = cv2.cvtColor(boost_for_detection(raw_frame),
                                   cv2.COLOR_BGR2RGB)
            result2 = self._hands_boost.process(rgb2)
            if result2.multi_hand_landmarks:
                self._missed_frames = 0
                return result2, display_frame

        return result, display_frame

    # ── Main loop ────────────────────────────────────────────────────────────

    def detect_gesture(self, get_frame):
        print(f"[GESTURE] Detection active (NV-fused) — Camera {self.cam_id}")

        while True:
            raw_frame = get_frame()
            if raw_frame is None:
                time.sleep(0.05)
                continue

            result, display_frame = self._run_hands(raw_frame)
            now        = time.time()
            hand_list  = []
            final_gest = None
            hint_text  = None

            if result.multi_hand_landmarks and result.multi_handedness:
                for lm, hd in zip(result.multi_hand_landmarks,
                                  result.multi_handedness):
                    label   = hd.classification[0].label
                    wrist_y = lm.landmark[0].y
                    hand_list.append((lm, label, wrist_y))
                    mp_draw.draw_landmarks(display_frame, lm,
                                           mp_hands.HAND_CONNECTIONS,
                                           _LM_SPEC, _CONN_SPEC)

            # ── 1. Raised open palm ──────────────────────────────
            raised_palms = [
                (lm, side) for lm, side, _ in hand_list
                if self._is_open_palm(lm, side) and self._arm_is_raised(lm)
            ]
            if raised_palms:
                if self._palm_start is None:
                    self._palm_start = now
                held  = now - self._palm_start
                label = ("BOTH HANDS RAISED — SOS"
                         if len(raised_palms) == 2
                         else f"OPEN PALM RAISED ({raised_palms[0][1]}) — SOS")
                if held >= PALM_HOLD_SECONDS:
                    final_gest = label
                else:
                    hint_text = f"HOLD PALM UP... {held:.1f}/{PALM_HOLD_SECONDS}s"
            else:
                self._palm_start = None

            # ── 2. Hand on heart ─────────────────────────────────
            if not final_gest and not raised_palms:
                heart_hands = [(lm, s) for lm, s, _ in hand_list
                               if self._hand_on_heart(lm)]
                if heart_hands:
                    if self._heart_start is None:
                        self._heart_start = now
                    held_h = now - self._heart_start
                    if held_h >= HEART_HOLD_SECONDS:
                        final_gest = f"HAND ON HEART ({heart_hands[0][1]}) — SOS"
                        self._heart_start = None
                    else:
                        hint_text = (hint_text or
                                     f"HOLD ON HEART... {held_h:.1f}/{HEART_HOLD_SECONDS}s")
                else:
                    self._heart_start = None

            # ── 3. Fist at chest ─────────────────────────────────
            if not final_gest and not raised_palms:
                if hand_list:
                    lm0, side0, wy0 = hand_list[0]
                    if self._is_fist(lm0, side0) and FIST_Y_MIN < wy0 < FIST_Y_MAX:
                        if self._fist_start is None:
                            self._fist_start = now
                        held_f = now - self._fist_start
                        if held_f >= FIST_HOLD_SECONDS:
                            final_gest = f"FIST AT CHEST ({side0}) — SOS"
                            self._fist_start = None
                        else:
                            hint_text = (hint_text or
                                         f"HOLD FIST... {held_f:.1f}/{FIST_HOLD_SECONDS}s")
                    else:
                        self._fist_start = None
                else:
                    self._fist_start = None

            # ── OSD ──────────────────────────────────────────────
            if is_dark_scene(raw_frame):
                _draw_bar(display_frame,
                          f"[NV-{_NV_MODE.upper()}]  CAM {self.cam_id}",
                          (0, 255, 0), y=30)

            display = final_gest or hint_text
            if display:
                colour = (0, 0, 255) if final_gest else (0, 165, 255)
                _draw_bar(display_frame, display[:60], colour, y=112)

            # ── Trigger ───────────────────────────────────────────
            if final_gest and (now - self.last_sos_time > self.cooldown):
                self._trigger_alert(final_gest, display_frame, get_frame)
                self.last_sos_time = now
                self._palm_start = self._heart_start = self._fist_start = None

            if self._set_anno is not None:
                self._set_anno(self.cam_id, display_frame)

            time.sleep(0.05)

    # ── Alert ─────────────────────────────────────────────────────────────────

    def _trigger_alert(self, gesture, snapshot, get_frame):
        print(f"[GESTURE] SOS confirmed — Camera {self.cam_id}: {gesture}")
        BASE_DIR   = os.getcwd()
        photo_path = os.path.join(BASE_DIR, f"gesture_cam{self.cam_id}.jpg")
        video_path = os.path.join(BASE_DIR, f"gesture_cam{self.cam_id}.mp4")
        nv_path    = os.path.join(BASE_DIR, f"gesture_nv_cam{self.cam_id}.mp4")
        cv2.imwrite(photo_path, snapshot)
        trigger_buzzer()
        threading.Thread(
            target=self._record_and_send,
            args=(get_frame, gesture, photo_path, video_path, nv_path),
            daemon=True,
        ).start()

    def _record_and_send(self, get_frame, gesture, photo_path, video_path, nv_path):
        print("[GESTURE] Recording main clip...")
        record_video_with_audio(get_frame, duration=5, filename=video_path)

        print("[GESTURE] Recording night-vision clip...")
        actual_nv = record_night_vision_clip(
            get_frame, duration=5, filename=nv_path,
            cam_id=self.cam_id, nv_mode=_NV_MODE,
        )
        time.sleep(3)

        print("[GESTURE] Sending to Telegram...")
        send_sos(
            message=f"Gesture SOS Detected!\n{gesture}",
            photo_path=photo_path,
            video_path=video_path,
            night_vision_path=actual_nv,
            cam_id=self.cam_id,
        )


# ── OSD helper ────────────────────────────────────────────────────────────────

def _draw_bar(frame, text: str, colour, y: int = 112):
    text_w, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)[0]
    cv2.rectangle(frame, (0, y - 24), (text_w + 20, y + 8), (0, 0, 0), -1)
    cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, colour, 2)


# ── Entry point ───────────────────────────────────────────────────────────────

def detect_gesture(get_frame, cam_id: int = 0, set_annotated_frame=None):
    GestureSOSDetector(
        cam_id=cam_id,
        set_annotated_frame=set_annotated_frame,
    ).detect_gesture(get_frame)
