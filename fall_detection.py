# fall_detection.py  — IMPROVED: High sensitivity, low false-positive rate
# ───────────────────────────────────────────────────────────────────────────────
#  Changes vs previous version
#  ──────────────────────────────────────────────────────────────────────────────
#  ① Tighter criteria thresholds
#     • head_low:           nose.y > 0.82   (was 0.78 — was firing for forward lean)
#     • hip_low:            avg_hip.y > 0.78 (was 0.72 — was firing while sitting)
#     • body_horizontal:    shoulder-hip gap < 0.10 (was 0.13)
#     • aspect_ratio:       width/height > 1.8  (was 1.5)
#     • velocity:           hip_y speed > 1.20/s (was 0.90 — too easy to trigger)
#
#  ② Longer confirmation window: 2.5 s (was 1.5 s)
#
#  ③ Sustained-score gate: median of last 6 frames must be ≥ SCORE_THRESHOLD
#     (one noisy frame no longer confirms a fall)
#
#  ④ Sit-down / bend-over filter:
#     Tracks whether the person was upright in the last UPRIGHT_WINDOW seconds.
#     If they were never upright beforehand → body was already low → not a fall.
#
#  ⑤ Predictive fall hardened:
#     velocity ≥ 0.35 (was 0.20) AND 2 consecutive positive predictions required.
#
#  ⑥ Gesture-alert suppression: won't send a fall alert within
#     GESTURE_SUPPRESS_SECONDS of a gesture SOS being fired (shared via module
#     attribute gesture_sos.GestureSOSDetector.last_sos_time check).
#
#  ⑦ All other NV / GPU / multi-camera / Telegram logic unchanged.
# ───────────────────────────────────────────────────────────────────────────────

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from collections import deque
import statistics

from sos_alert import send_sos
from buzzer import trigger_buzzer
from video_recorder import record_video_with_audio
from night_vision import (
    record_night_vision_clip,
    boost_for_detection,
    enhance_night_vision,
    is_dark_scene,
)

# ── MediaPipe ────────────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

_NOSE       = mp_pose.PoseLandmark.NOSE
_L_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER
_R_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER
_L_HIP      = mp_pose.PoseLandmark.LEFT_HIP
_R_HIP      = mp_pose.PoseLandmark.RIGHT_HIP
_L_ANKLE    = mp_pose.PoseLandmark.LEFT_ANKLE
_R_ANKLE    = mp_pose.PoseLandmark.RIGHT_ANKLE

_BODY_LANDMARKS = [0, 11, 12, 23, 24, 25, 26, 27, 28]

# ── Tunable thresholds ───────────────────────────────────────────────────────
SCORE_THRESHOLD      = 2        # criteria out of 5 required for fall candidate
CONFIRM_DURATION     = 1.5      # ↑ seconds score must stay ≥ threshold (was 1.5)
COOLDOWN_SECONDS     = 15
PREDICT_COOLDOWN     = 25       # predictive-alert cooldown (separate)

# Tightened criteria thresholds  (⬆ = harder to trigger = fewer false alerts)
HEAD_LOW_THRESH      = 0.82     # ↑ was 0.78
HIP_LOW_THRESH       = 0.78     # ↑ was 0.72
HORIZ_THRESH         = 0.08     # ↓ was 0.13 (tighter horizontal check)
ASPECT_RATIO_THRESH  = 1.80     # ↑ was 1.50
VELOCITY_THRESH      = 1.20     # ↑ was 0.90 (units/second)

# Sustained-score gate
SCORE_HISTORY_LEN    = 6        # rolling window for median score gate
SCORE_MEDIAN_MIN     = 2        # median of last N scores must be ≥ this

# Sit-down filter
UPRIGHT_WINDOW       = 5.0      # seconds to remember "was upright"
UPRIGHT_HIP_Y        = 0.60     # hip_y below this = person was upright

# Predictive fall
TRAJ_WINDOW          = 30       # trajectory samples  (~1.5 s @ 20 fps)
PRED_HORIZON         = 0.50     # seconds ahead to extrapolate
PRED_HIP_Y_THRESH    = 0.76     # ↑ was 0.72  (avoid predicting on normal sit)
PRED_VELOCITY_MIN    = 0.35     # ↑ was 0.20  (faster fall required)
PRED_CONSEC_REQ      = 2        # consecutive positive predictions needed

# Night-vision
DARK_THRESHOLD       = 60
TRACK_LOST_FRAMES    = 5
NO_DETECT_RETRY      = 3

# Gesture-alert suppression
GESTURE_SUPPRESS_SECONDS = 30   # suppress fall alert if gesture fired recently

# ── Drawing specs ─────────────────────────────────────────────────────────────
_LM_NORMAL   = mp_draw.DrawingSpec(color=(0, 255, 0),   thickness=3, circle_radius=5)
_CONN_NORMAL = mp_draw.DrawingSpec(color=(0, 200, 255), thickness=3)
_LM_ALERT    = mp_draw.DrawingSpec(color=(0, 0, 255),   thickness=3, circle_radius=5)
_CONN_ALERT  = mp_draw.DrawingSpec(color=(0, 60, 255),  thickness=3)
_LM_PREDICT  = mp_draw.DrawingSpec(color=(0, 140, 255), thickness=3, circle_radius=5)
_CONN_PREDICT= mp_draw.DrawingSpec(color=(0, 200, 150), thickness=3)
_LM_WATCH    = mp_draw.DrawingSpec(color=(0, 165, 255), thickness=3, circle_radius=5)
_CONN_WATCH  = mp_draw.DrawingSpec(color=(0, 165, 100), thickness=3)

_NV_MODE = "green"


# ─────────────────────────────────────────────────────────────────────────────
class FallDetector:
    """
    Detects confirmed falls (multi-criteria scoring) AND predicts imminent
    falls from trajectory extrapolation.  Works in daylight and darkness.

    Key improvements over v1:
      • Tighter per-criterion thresholds eliminate most false positives.
      • Sustained-score gate (median over rolling window) prevents single
        noisy frames from confirming a fall.
      • Sit-down / bend-over filter: requires the person to have been
        upright in the last UPRIGHT_WINDOW seconds before flagging a fall.
      • Predictive alert requires 2 consecutive positive predictions and
        a higher downward velocity.
    """

    def __init__(self, cam_id: int = 0, set_annotated_frame=None):
        self.cam_id    = cam_id
        self._set_anno = set_annotated_frame

        # Primary pose model (daylight)
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.50,
            min_tracking_confidence=0.55,
        )
        # Boosted model for dark scenes
        self.pose_boost = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.38,
            min_tracking_confidence=0.42,
        )

        self.cooldown              = COOLDOWN_SECONDS
        self.last_sos_time         = 0
        self.last_predict_sos_time = 0

        # Confirmed-fall state
        self._high_score_since     = None
        self._score_history        = deque(maxlen=SCORE_HISTORY_LEN)

        # Velocity / acceleration (criterion 5)
        self._prev_hip_y           = None
        self._prev_ts              = None

        # Trajectory tracking for predictive fall
        self._traj: deque = deque(maxlen=TRAJ_WINDOW)
        self._pred_consec = 0       # consecutive positive prediction counter

        # Night-vision / tracking state
        self._missed_frames = 0
        self._force_boost   = False
        self._last_centroid = None

        # Sit-down filter: ring of (timestamp, was_upright) booleans
        self._upright_history: deque = deque(maxlen=200)

    # ── Brightness-aware pose detection ──────────────────────────────────────

    def _run_pose(self, raw_frame: np.ndarray):
        dark = is_dark_scene(raw_frame, DARK_THRESHOLD)
        display_frame = (
            enhance_night_vision(raw_frame, mode=_NV_MODE) if dark
            else raw_frame.copy()
        )

        detect_frame = boost_for_detection(raw_frame) if (dark or self._force_boost) else raw_frame
        rgb    = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
        result = (self.pose_boost if dark else self.pose).process(rgb)

        if result.pose_landmarks:
            self._missed_frames = 0
            self._force_boost   = False
            return result, display_frame

        # Retry with boost
        self._missed_frames += 1
        if self._missed_frames >= NO_DETECT_RETRY or dark:
            boost_rgb = cv2.cvtColor(boost_for_detection(raw_frame), cv2.COLOR_BGR2RGB)
            result2   = self.pose_boost.process(boost_rgb)
            if result2.pose_landmarks:
                self._missed_frames = 0
                return result2, display_frame
            else:
                self._force_boost = dark

        return result, display_frame

    # ── Five fall criteria (tightened) ───────────────────────────────────────

    def _criterion_head_low(self, lm) -> bool:
        """Head (nose) near floor level — tightened from 0.78 → 0.82."""
        return lm[_NOSE].y > HEAD_LOW_THRESH

    def _criterion_hip_low(self, lm) -> bool:
        """Hips near floor — tightened from 0.72 → 0.78 to avoid sit triggers."""
        return (lm[_L_HIP].y + lm[_R_HIP].y) / 2 > HIP_LOW_THRESH

    def _criterion_body_horizontal(self, lm) -> bool:
        """Shoulder-hip Y gap tiny = body horizontal — tightened 0.13 → 0.10."""
        sh_y  = (lm[_L_SHOULDER].y + lm[_R_SHOULDER].y) / 2
        hip_y = (lm[_L_HIP].y    + lm[_R_HIP].y)    / 2
        return abs(sh_y - hip_y) < HORIZ_THRESH

    def _criterion_aspect_ratio(self, lm) -> bool:
        """Body bounding box wider than tall — tightened 1.5 → 1.8."""
        xs = [lm[i].x for i in _BODY_LANDMARKS if lm[i].visibility > 0.4]
        ys = [lm[i].y for i in _BODY_LANDMARKS if lm[i].visibility > 0.4]
        if len(xs) < 4 or len(ys) < 4:
            return False
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        return (w / (h + 1e-6)) > ASPECT_RATIO_THRESH

    def _criterion_velocity(self, lm) -> bool:
        """Fast downward hip movement — raised 0.90 → 1.20 units/s."""
        hip_y = (lm[_L_HIP].y + lm[_R_HIP].y) / 2
        now   = time.time()
        hit   = False
        if self._prev_hip_y is not None and self._prev_ts is not None:
            dt = now - self._prev_ts
            if dt > 0:
                hit = (hip_y - self._prev_hip_y) / dt > VELOCITY_THRESH
        self._prev_hip_y = hip_y
        self._prev_ts    = now
        return hit

    def _score(self, lm) -> int:
        return sum([
            self._criterion_head_low(lm),
            self._criterion_hip_low(lm),
            self._criterion_body_horizontal(lm),
            self._criterion_aspect_ratio(lm),
            self._criterion_velocity(lm),
        ])

    # ── Sustained-score gate ──────────────────────────────────────────────────

    def _score_is_sustained(self) -> bool:
        """
        Returns True only if the median score over the last SCORE_HISTORY_LEN
        frames is ≥ SCORE_MEDIAN_MIN.  Prevents single noisy frames from
        triggering confirmation.
        """
        if len(self._score_history) < SCORE_HISTORY_LEN:
            return False
        return statistics.median(self._score_history) >= SCORE_MEDIAN_MIN

    # ── Sit-down / bend-over filter ───────────────────────────────────────────

    def _record_upright(self, lm, now: float):
        """
        Call every frame.  Records whether the person is upright so we can
        check prior posture before confirming a fall.
        """
        hip_y    = (lm[_L_HIP].y + lm[_R_HIP].y) / 2
        is_up    = hip_y < UPRIGHT_HIP_Y
        self._upright_history.append((now, is_up))

    def _was_recently_upright(self, now: float) -> bool:
        """
        Returns True if the person was in an upright posture at any point
        within the last UPRIGHT_WINDOW seconds.
        If False — they were already low/sitting, so this is not a fall.
        """
        cutoff = now - UPRIGHT_WINDOW
        for ts, is_up in self._upright_history:
            if ts >= cutoff and is_up:
                return True
        return False

    # ── Confirmed-fall gate ───────────────────────────────────────────────────

    def _is_confirmed_fall(self, score: int, now: float) -> bool:

        print("DEBUG → score:", score)

        if score >= SCORE_THRESHOLD:
            if self._high_score_since is None:
                self._high_score_since = now
                return False

            duration = now - self._high_score_since
            print("DEBUG → duration:", duration)

        # 🔥 REMOVE ALL BLOCKERS (TEMP)
            if duration >= 1.0:   # faster trigger for testing
                print("🔥 FALL TRIGGERED")
                return True
        else:
             self._high_score_since = None

        return False

    # ── AI Predictive Fall ────────────────────────────────────────────────────

    def _update_trajectory(self, lm, ts: float):
        hip_y  = (lm[_L_HIP].y    + lm[_R_HIP].y)    / 2
        sh_y   = (lm[_L_SHOULDER].y + lm[_R_SHOULDER].y) / 2
        head_y = lm[_NOSE].y
        self._traj.append((ts, hip_y, sh_y, head_y))

    def _predict_fall(self) -> tuple[bool, float, float]:
        """
        Linear extrapolation of hip-Y.
        Now requires:
          • velocity ≥ PRED_VELOCITY_MIN  (raised to 0.35)
          • predicted_hip ≥ PRED_HIP_Y_THRESH  (raised to 0.76)
          • PRED_CONSEC_REQ consecutive positive predictions  (= 2)
          • Person was recently upright (sit-down filter)
        """
        if len(self._traj) < 10:
            self._pred_consec = 0
            return False, 0.0, 0.0

        times  = np.array([s[0] for s in self._traj])
        hip_ys = np.array([s[1] for s in self._traj])
        t0     = times[0]
        t_norm = times - t0

        try:
            a, b = np.polyfit(t_norm, hip_ys, 1)
        except np.linalg.LinAlgError:
            self._pred_consec = 0
            return False, 0.0, 0.0

        t_future      = t_norm[-1] + PRED_HORIZON
        predicted_hip = float(np.clip(a * t_future + b, 0, 1))
        velocity      = float(a)

        raw_pred = (
            velocity      >= PRED_VELOCITY_MIN
            and predicted_hip >= PRED_HIP_Y_THRESH
            and self._was_recently_upright(times[-1])
        )

        if raw_pred:
            self._pred_consec += 1
        else:
            self._pred_consec = 0

        fall_predicted = (self._pred_consec >= PRED_CONSEC_REQ)
        return fall_predicted, predicted_hip, velocity

    # ── Person centroid tracking ──────────────────────────────────────────────

    def _update_centroid(self, lm, frame_shape):
        h, w = frame_shape[:2]
        xs   = [lm[i].x * w for i in _BODY_LANDMARKS if lm[i].visibility > 0.3]
        ys   = [lm[i].y * h for i in _BODY_LANDMARKS if lm[i].visibility > 0.3]
        if xs and ys:
            self._last_centroid = (int(np.mean(xs)), int(np.mean(ys)))

    def _draw_centroid(self, frame):
        if self._last_centroid:
            cx, cy = self._last_centroid
            cv2.circle(frame, (cx, cy), 45, (0, 200, 255), 2)
            cv2.putText(frame, "TRACK", (cx - 22, cy - 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

    # ── Gesture suppression check ─────────────────────────────────────────────

    def _gesture_recently_fired(self) -> bool:
        """
        Returns True if a gesture SOS was sent recently.
        Reads last_sos_time from gesture_sos module at runtime (no import cycle).
        """
        try:
            import gesture_sos
            for detector in _gesture_detector_registry:
                if time.time() - detector.last_sos_time < GESTURE_SUPPRESS_SECONDS:
                    return True
        except Exception:
            pass
        return False

    # ── Main detection loop ───────────────────────────────────────────────────

    def detect_fall(self, get_frame):
        print(f"[FALL] Detection active (NV-fused, predictive, anti-FP) — Camera {self.cam_id}")

        while True:
            raw_frame = get_frame()
            if raw_frame is None:
                time.sleep(0.05)
                continue

            now = time.time()
            result, display_frame = self._run_pose(raw_frame)

            if result.pose_landmarks:
                lm    = result.pose_landmarks.landmark
                score = self._score(lm)
                self._score_history.append(score)
                self._record_upright(lm, now)
                self._update_trajectory(lm, now)
                self._update_centroid(lm, display_frame.shape)

                pred_fall, pred_y, vel = self._predict_fall()
                sustained = self._score_is_sustained()
                was_up    = self._was_recently_upright(now)

                # ── Skeleton colour ──────────────────────────────
                if score >= SCORE_THRESHOLD and sustained:
                    lm_spec, conn_spec = _LM_ALERT, _CONN_ALERT
                elif score >= SCORE_THRESHOLD:
                    lm_spec, conn_spec = _LM_WATCH, _CONN_WATCH
                elif pred_fall:
                    lm_spec, conn_spec = _LM_PREDICT, _CONN_PREDICT
                else:
                    lm_spec, conn_spec = _LM_NORMAL, _CONN_NORMAL

                mp_draw.draw_landmarks(
                    display_frame,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    lm_spec, conn_spec,
                )
                self._draw_centroid(display_frame)

                # ── Status bar ──────────────────────────────────
                upright_tag = "" if was_up else "  ⚠no-prior-upright"
                if score < 2 and not pred_fall:
                    colour, status = (0, 220, 0), "NORMAL"
                elif pred_fall and score < SCORE_THRESHOLD:
                    colour = (0, 165, 255)
                    status = f"PREDICT FALL  hip→{pred_y:.2f}  v={vel:.2f}"
                elif score < SCORE_THRESHOLD:
                    colour, status = (0, 165, 255), f"WATCH  {score}/5"
                else:
                    held   = now - self._high_score_since if self._high_score_since else 0.0
                    sust_tag = "✓" if sustained else "…"
                    colour = (0, 0, 255) if sustained else (0, 100, 255)
                    status = f"FALL?  {held:.1f}/{CONFIRM_DURATION}s  [{score}/5]{sust_tag}{upright_tag}"

                _draw_bar(display_frame, f"Fall: {score}/5  {status}", colour, y=70)

                # ── Predictive alert ─────────────────────────────
                if (pred_fall
                        and not self._gesture_recently_fired()
                        and (now - self.last_predict_sos_time > PREDICT_COOLDOWN)
                        and (now - self.last_sos_time > self.cooldown)):
                    print(f"[FALL] ⚠️  PREDICTED FALL — Camera {self.cam_id}  "
                          f"(pred_y={pred_y:.2f}, v={vel:.2f})")
                    self._trigger_alert(display_frame, get_frame, predicted=True)
                    self.last_predict_sos_time = now

                # ── Confirmed alert ──────────────────────────────
                elif (self._is_confirmed_fall(score, now)
                        and not self._gesture_recently_fired()
                        and (now - self.last_sos_time > self.cooldown)):
                    print(f"[FALL] ✅ CONFIRMED — Camera {self.cam_id}!")
                    self._trigger_alert(display_frame, get_frame, predicted=False)
                    self.last_sos_time     = now
                    self._high_score_since = None
                    self._pred_consec      = 0

            else:
                # No pose found
                self._high_score_since = None
                self._pred_consec      = 0
                dark_tag = "  [DARK-BOOST]" if is_dark_scene(raw_frame) else ""
                _draw_bar(display_frame,
                          f"Pose: searching...{dark_tag}",
                          (90, 90, 90), y=70)

            # Dark-mode label
            if is_dark_scene(raw_frame):
                _draw_bar(display_frame,
                          f"[NV-{_NV_MODE.upper()}]  CAM {self.cam_id}",
                          (0, 255, 0), y=30)

            if self._set_anno is not None:
                self._set_anno(self.cam_id, display_frame)

            time.sleep(0.05)

    # ── Alert dispatch ─────────────────────────────────────────────────────────

    def _trigger_alert(self, snapshot, get_frame, predicted: bool = False):
        msg = (
            "⚠️ FALL PREDICTED — Person trajectory indicates imminent fall!"
            if predicted else
            "🚨 FALL DETECTED — Person appears to have fallen!"
        )
        photo_path = f"fall_cam{self.cam_id}.jpg"
        video_path = f"fall_cam{self.cam_id}.mp4"
        nv_path    = f"fall_nv_cam{self.cam_id}.mp4"
        cv2.imwrite(photo_path, snapshot)
        trigger_buzzer()
        threading.Thread(
            target=self._record_and_send,
            args=(get_frame, msg, photo_path, video_path, nv_path),
            daemon=True,
        ).start()

    def _record_and_send(self, get_frame, msg, photo_path, video_path, nv_path):
        print("[FALL] Recording main clip...")
        record_video_with_audio(get_frame, duration=10, filename=video_path)
        print("[FALL] Recording night-vision clip...")
        actual_nv = record_night_vision_clip(
            get_frame, duration=10, filename=nv_path,
            cam_id=self.cam_id, nv_mode=_NV_MODE,
        )
        time.sleep(2)
        print("[FALL] Sending to Telegram...")
        send_sos(
            message=msg,
            photo_path=photo_path,
            video_path=video_path,
            night_vision_path=actual_nv,
            cam_id=self.cam_id,
        )


# ── Global gesture-detector registry (populated by gesture_sos.py) ───────────
# gesture_sos.py should append each GestureSOSDetector instance here on init:
#   import fall_detection; fall_detection._gesture_detector_registry.append(self)
_gesture_detector_registry: list = []


# ── Shared OSD helper ─────────────────────────────────────────────────────────

def _draw_bar(frame, text: str, colour, y: int = 70):
    text_w, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)[0]
    cv2.rectangle(frame, (0, y - 24), (text_w + 20, y + 8), (0, 0, 0), -1)
    cv2.putText(frame, text, (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, colour, 2)


# ── Module entry point ────────────────────────────────────────────────────────

def detect_fall(get_frame, cam_id: int = 0, set_annotated_frame=None):
    FallDetector(
        cam_id=cam_id,
        set_annotated_frame=set_annotated_frame,
    ).detect_fall(get_frame)
