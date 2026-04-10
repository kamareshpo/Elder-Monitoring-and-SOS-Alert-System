# buzzer.py  — Windows built-in winsound (no pygame needed)
import threading
import os


def trigger_buzzer(beeps: int = 5, freq_hz: int = 1200, beep_ms: int = 400):
    """Non-blocking multi-beep alarm using winsound (Windows built-in, zero dependencies)."""
    threading.Thread(target=_play_alarm,
                     args=(beeps, freq_hz, beep_ms), daemon=True).start()


def _play_alarm(beeps: int, freq_hz: int, beep_ms: int):
    try:
        import winsound
        for _ in range(beeps):
            winsound.Beep(freq_hz, beep_ms)
            threading.Event().wait(0.15)   # 150 ms gap
    except Exception as e:
        for _ in range(beeps):
            print("\a", end="", flush=True)
        print(f"🔔 BUZZER ALERT  ({e})")


def generate_telegram_beep(out_path: str = "telegram_beep.ogg",
                            beeps: int = 5,
                            freq_hz: int = 1000,
                            beep_ms: int = 400):
    """
    Generates a repeating beep .ogg voice note for Telegram.
    Requires ffmpeg in PATH.  Returns path on success, None on failure.
    """
    try:
        import numpy as np
        import scipy.io.wavfile as wav
        import subprocess

        sample_rate = 44100
        beep_s      = int(sample_rate * beep_ms / 1000)
        gap_s       = int(sample_rate * 0.15)
        t_beep      = __import__('numpy').linspace(0, beep_ms / 1000, beep_s, endpoint=False)
        beep        = __import__('numpy').int16(
                          __import__('numpy').sin(2 * 3.14159 * freq_hz * t_beep) * 32767)
        gap         = __import__('numpy').zeros(gap_s, dtype=__import__('numpy').int16)
        signal      = __import__('numpy').concatenate(
                          [__import__('numpy').concatenate([beep, gap]) for _ in range(beeps)])

        wav_path = out_path.replace(".ogg", "_tmp.wav")
        wav.write(wav_path, sample_rate, signal)

        result = subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-c:a", "libopus", out_path],
            capture_output=True, timeout=15
        )
        if os.path.exists(wav_path):
            os.remove(wav_path)

        return out_path if (result.returncode == 0 and os.path.exists(out_path)) else None

    except FileNotFoundError:
        print("⚠️  ffmpeg not found — Telegram beep skipped. Install ffmpeg: winget install ffmpeg")
        return None
    except Exception as e:
        print(f"⚠️  Could not generate Telegram beep: {e}")
        return None