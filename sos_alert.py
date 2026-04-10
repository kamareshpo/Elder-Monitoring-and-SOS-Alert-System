# sos_alert.py — FINAL STABLE VERSION (No spam + Reliable media send + Alarm)

import requests
import os
import time
from datetime import datetime

# ── Load credentials ─────────────────────────────────────────────
try:
    from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
except ImportError:
    TELEGRAM_TOKEN = "YOUR_BOT_TOKEN"
    TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# 🔥 MacroDroid Webhook URL
MACRODROID_URL = "https://trigger.macrodroid.com/d4c1eaf0-13ed-4559-92e2-8a1e21a6fbfc/alert"


# ── Telegram POST ────────────────────────────────────────────────
def _post(endpoint: str, data: dict, files: dict | None = None):
    try:
        r = requests.post(
            f"{BASE_URL}/{endpoint}",
            data=data,
            files=files,
            timeout=60
        )
        print(f"[Telegram {endpoint}] → {r.status_code} | {r.text}")
        return r
    except requests.RequestException as e:
        print(f"⚠️ Telegram {endpoint} failed:", e)

# 🔊 Trigger phone alarm (MacroDroid)
def trigger_phone_alarm():
    try:
        requests.get(MACRODROID_URL, timeout=3)
        print("🔊 PHONE ALARM TRIGGERED")
    except Exception as e:
        print("⚠️ Alarm trigger failed:", e)


# ⏳ Wait for file to be ready
def wait_for_file(path, timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        if path and os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"✅ File ready: {path}")
            return True
        time.sleep(0.5)
    print(f"❌ File NOT ready: {path}")
    return False


# ── MAIN ALERT FUNCTION ──────────────────────────────────────────
def send_sos(
    message: str,
    photo_path: str | None = None,
    video_path: str | None = None,
    night_vision_path: str | None = None,
    cam_id: int = 0,
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    full_msg = (
        f"🚨 ALERT — Camera {cam_id}\n"
        f"{message}\n\n"
        f"Time: {timestamp}"
    )

    # ✅ 1. Send ONE clean message
    _post("sendMessage", {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": full_msg,
    })

    print("📤 Alert message sent")

    # 🔥 2. Trigger phone alarm
    trigger_phone_alarm()

    # ⏳ 3. Wait before sending files
    time.sleep(2)

    # ── PHOTO ────────────────────────────────────────────────
    if photo_path and wait_for_file(photo_path):
        try:
            print("📸 Sending photo:", photo_path)
            with open(photo_path, "rb") as f:
                _post("sendPhoto", {
                    "chat_id": TELEGRAM_CHAT_ID,
                    "caption": f"Snapshot — Cam {cam_id}",
                }, files={"photo": f})
        except Exception as e:
            print("⚠️ Photo send failed:", e)
    else:
        print("⚠️ Photo skipped")

    # ── VIDEO ────────────────────────────────────────────────
    if video_path and wait_for_file(video_path, timeout=20):
        try:
            print("🎥 Sending video:", video_path)
            with open(video_path, "rb") as f:
                _post("sendVideo", {
                    "chat_id": TELEGRAM_CHAT_ID,
                    "caption": f"Alert clip — Cam {cam_id}",
                    "supports_streaming": "true",
                }, files={"video": f})
        except Exception as e:
            print("⚠️ Video send failed:", e)
    else:
        print("⚠️ Video skipped")

    # ── NIGHT VISION ─────────────────────────────────────────
    if night_vision_path and wait_for_file(night_vision_path, timeout=20):
        try:
            print("🌙 Sending night vision:", night_vision_path)
            with open(night_vision_path, "rb") as f:
                _post("sendVideo", {
                    "chat_id": TELEGRAM_CHAT_ID,
                    "caption": f"Night Vision — Cam {cam_id}",
                }, files={"video": f})
        except Exception as e:
            print("⚠️ Night vision send failed:", e)
    else:
        print("⚠️ Night vision skipped")

    # ── OPTIONAL BUZZER AUDIO ────────────────────────────────
    try:
        from buzzer import generate_telegram_beep

        beep_file = generate_telegram_beep("telegram_beep.ogg")

        if beep_file and os.path.exists(beep_file):
            print("🔔 Sending alert sound")

            with open(beep_file, "rb") as f:
                _post("sendVoice", {
                    "chat_id": TELEGRAM_CHAT_ID,
                    "caption": "🔔 RESPOND IMMEDIATELY!",
                }, files={"voice": f})

            os.remove(beep_file)

    except Exception as e:
        print("⚠️ Beep send failed:", e)