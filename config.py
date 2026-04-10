# utils/config.py  — Credentials and tunable settings
# ─────────────────────────────────────────────────────────────────
#  ⚠  NEVER commit this file to a public repository.
#     Add utils/config.py to your .gitignore.
# ─────────────────────────────────────────────────────────────────

# ── Telegram ──────────────────────────────────────────────────────
TELEGRAM_TOKEN   = ""     # e.g. "7899915167:AAFWxl9J..."
TELEGRAM_CHAT_ID = ""       # e.g. "1114449056"

# ── Detection thresholds ──────────────────────────────────────────
NO_MOVEMENT_THRESHOLD_HOURS = 1         # hours before no-motion alert
FALL_COOLDOWN_SECONDS       = 15        # min gap between fall alerts
GESTURE_COOLDOWN_SECONDS    = 15        # min gap between gesture alerts

# ── Camera indices ────────────────────────────────────────────────
CAMERA_IDS = [0, 1, 2]                  # adjust to your hardware
