# video_recorder.py  — Records video frames + microphone audio, merges with ffmpeg
import cv2
import threading
import time
import os
import subprocess
import tempfile


def record_video_with_audio(
    get_frame,
    duration: int = 10,
    filename: str = "alert_clip.mp4",
    fps: float = 20.0,
) -> str:
    """
    Records `duration` seconds of:
      • video from the shared camera frame
      • audio from the default system microphone

    Merges both with ffmpeg → returns the output filename.
    Falls back to silent video if audio capture fails.
    """
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)

    tmp_video = filename + ".tmp.avi"
    tmp_audio = filename + ".tmp.wav"

    # ── Video recording thread ───────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(tmp_video, fourcc, fps, (640, 480))
    video_done = threading.Event()

    def write_video():
        start = time.time()
        while time.time() - start < duration:
            frame = get_frame()
            if frame is not None:
                out.write(cv2.resize(frame, (640, 480)))
            time.sleep(1 / fps)
        out.release()
        video_done.set()

    # ── Audio recording thread (sounddevice) ────────────────────────────────
    audio_ok = False
    try:
        import sounddevice as sd
        import scipy.io.wavfile as wav
        import numpy as np

        SAMPLE_RATE = 44100
        audio_frames = []
        audio_done = threading.Event()

        def write_audio():
            nonlocal audio_ok
            try:
                recording = sd.rec(
                    int(duration * SAMPLE_RATE),
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="int16",
                )
                sd.wait()
                wav.write(tmp_audio, SAMPLE_RATE, recording)
                audio_ok = True
            except Exception as e:
                print(f"⚠️  Audio capture failed: {e}")
            audio_done.set()

        audio_thread = threading.Thread(target=write_audio, daemon=True)
        audio_thread.start()
    except ImportError:
        print("⚠️  sounddevice not installed — recording video only.")

    vid_thread = threading.Thread(target=write_video, daemon=True)
    vid_thread.start()
    vid_thread.join(timeout=duration + 5)

    # ── Merge with ffmpeg ────────────────────────────────────────────────────
    try:
        if audio_ok and os.path.exists(tmp_audio):
            cmd = [
                "ffmpeg", "-y",
                "-i", tmp_video,
                "-i", tmp_audio,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-shortest",
                filename,
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", tmp_video,
                "-c:v", "libx264",
                filename,
            ]
        subprocess.run(cmd, capture_output=True, timeout=30)
    except Exception as e:
        print(f"⚠️  ffmpeg merge failed ({e}), using raw AVI.")
        filename = tmp_video
        tmp_video = None   # don't delete the fallback

    # Cleanup temp files
    for f in [tmp_video, tmp_audio]:
        if f and os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass

    return filename
