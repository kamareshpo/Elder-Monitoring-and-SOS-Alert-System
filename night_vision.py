# night_vision.py — GPU-accelerated | Thermal IR | Crystal-clear NV | Predictive boost
# ───────────────────────────────────────────────────────────────────────────────
#  Modes:
#    'green'   — Classic NV green phosphor (default, clean & sharp)
#    'thermal' — Real IR thermal simulation  (INFERNO heat-map colormap)
#    'white'   — White phosphor NV  (modern military / Gen-3 image intensifier)
#
#  GPU:  Automatically uses CUDA CLAHE + bilateral filter when a compatible
#        NVIDIA GPU is present.  Falls back to CPU seamlessly.
#
#  boost_for_detection() — super-sharpen for MediaPipe in complete darkness.
# ───────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
import time
import os
import subprocess

# ── GPU auto-detect ──────────────────────────────────────────────────────────
_GPU = False
_gpu_clahe = None

try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        _gpu_clahe = cv2.cuda.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        _GPU = True
        print("[NV] ✅ CUDA GPU detected — GPU acceleration ENABLED")
    else:
        print("[NV] No CUDA GPU — CPU mode")
except Exception:
    print("[NV] CUDA not available — CPU mode")

# ── Shared CPU CLAHE (always available as fallback) ──────────────────────────
_clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))

# Gamma LUT  (γ = 2.2 — natural luminance boost, avoids blown-out whites)
_GAMMA     = 2.2
_gamma_lut = np.array(
    [(i / 255.0) ** (1.0 / _GAMMA) * 255 for i in range(256)], dtype=np.uint8
)

# Sharpening kernel  (mild: avoids noise amplification)
_SHARP_K = np.array([
    [ 0, -0.5,  0],
    [-0.5,  3, -0.5],
    [ 0, -0.5,  0],
], dtype=np.float32)

# Stronger kernel for boost_for_detection()
_BOOST_K = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1],
], dtype=np.float32)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _clahe_gpu(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE on GPU if available, else CPU."""
    if _GPU and _gpu_clahe is not None:
        try:
            g  = cv2.cuda_GpuMat(); g.upload(gray)
            r  = _gpu_clahe.apply(g, cv2.cuda_Stream.Null())
            return r.download()
        except Exception:
            pass
    return _clahe.apply(gray)


def _bilateral_gpu(gray: np.ndarray, d: int = 9,
                   sc: float = 80, ss: float = 80) -> np.ndarray:
    """
    Edge-preserving denoise.
    GPU path: cv2.cuda.bilateralFilter
    CPU path: cv2.bilateralFilter
    Both preserve joint/limb boundaries far better than Gaussian.
    """
    if _GPU:
        try:
            g = cv2.cuda_GpuMat(); g.upload(gray)
            r = cv2.cuda.bilateralFilter(g, d, sc, ss)
            return r.download()
        except Exception:
            pass
    return cv2.bilateralFilter(gray, d=d, sigmaColor=sc, sigmaSpace=ss)


def _enhance_luma(y: np.ndarray, strong: bool = False) -> np.ndarray:
    """Full luma pipeline: CLAHE → gamma → bilateral → sharpen."""
    y = _clahe_gpu(y)
    y = cv2.LUT(y, _gamma_lut)
    d, sc = (7, 60) if strong else (9, 80)
    y = _bilateral_gpu(y, d=d, sc=sc, ss=sc)
    kern = _BOOST_K if strong else _SHARP_K
    y_f  = cv2.filter2D(y.astype(np.float32), -1, kern)
    return np.clip(y_f, 0, 255).astype(np.uint8)


# ── Scene brightness check ───────────────────────────────────────────────────

def is_dark_scene(frame: np.ndarray, threshold: int = 55) -> bool:
    """Return True when mean luminance is below threshold (≈ nighttime)."""
    if frame is None:
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) < threshold


# ── Main NV enhancement ──────────────────────────────────────────────────────

def enhance_night_vision(
    frame: np.ndarray,
    mode: str = "green",
) -> np.ndarray:
    """
    Returns a crystal-clear night-vision frame.

    Parameters
    ----------
    frame : BGR input frame
    mode  : 'green' | 'thermal' | 'white'

    'green'   — Gen-2 green phosphor NV tube simulation
    'thermal' — LWIR thermal imaging simulation  (cold=dark, hot=bright)
    'white'   — Gen-3 white phosphor (cleaner, tactical)
    """
    if frame is None:
        return frame

    ycrcb        = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb    = cv2.split(ycrcb)
    y            = _enhance_luma(y, strong=False)

    # ── Thermal / IR mode ──────────────────────────────────────────
    if mode == "thermal":
        # Simulate skin/warm-body emission: slightly push warm tones up
        # before applying inferno colormap so bodies glow orange-white
        y_warm = np.clip(y.astype(np.int16) + cr.astype(np.int16) // 6,
                         0, 255).astype(np.uint8)
        thermal = cv2.applyColorMap(y_warm, cv2.COLORMAP_INFERNO)

        # Authentic faint scan-line texture
        for row in range(0, thermal.shape[0], 3):
            thermal[row] = np.clip(
                thermal[row].astype(np.int16) - 12, 0, 255
            ).astype(np.uint8)

        # Slight gaussian to soften pixelation
        thermal = cv2.GaussianBlur(thermal, (3, 3), 0)
        return thermal

    # ── White phosphor mode ────────────────────────────────────────
    elif mode == "white":
        # Extra contrast stretch for bright-white look
        y_w = cv2.normalize(y, None, alpha=30, beta=255,
                            norm_type=cv2.NORM_MINMAX)
        # Convert to BGR grey stack
        white_frame = cv2.cvtColor(y_w, cv2.COLOR_GRAY2BGR)
        # Slight cool-blue tint (period accurate)
        b, g, r = cv2.split(white_frame)
        b = np.clip(b.astype(np.int16) + 15, 0, 255).astype(np.uint8)
        return cv2.merge([b, g, r])

    # ── Green phosphor mode (default) ─────────────────────────────
    else:
        enhanced    = cv2.cvtColor(cv2.merge([y, cr, cb]),
                                   cv2.COLOR_YCrCb2BGR)
        b, g, r     = cv2.split(enhanced)
        g_boost     = np.clip(g.astype(np.int16) + 25, 0, 255).astype(np.uint8)
        night_frame = cv2.merge([
            (b.astype(np.uint16) * 25 // 255).astype(np.uint8),
            g_boost,
            (r.astype(np.uint16) * 25 // 255).astype(np.uint8),
        ])
        return night_frame


# ── Detection boost  (for MediaPipe in the dark) ─────────────────────────────

def boost_for_detection(frame: np.ndarray) -> np.ndarray:
    """
    Maximally brighten & sharpen a frame so MediaPipe can find
    pose/hand landmarks even in near-darkness.

    Returns clean BGR (no colour tint) — draw NV overlay separately.
    """
    if frame is None:
        return frame

    ycrcb        = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb    = cv2.split(ycrcb)
    y            = _enhance_luma(y, strong=True)      # stronger CLAHE + kernel

    # Extra brightness clip-stretch
    y = cv2.normalize(y, None, alpha=20, beta=255, norm_type=cv2.NORM_MINMAX)

    boosted = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)

    # Unsharp mask: makes every edge (limb, hand) crisper for landmark detection
    blurred   = cv2.GaussianBlur(boosted, (0, 0), 3)
    sharpened = cv2.addWeighted(boosted, 1.9, blurred, -0.9, 0)
    return sharpened


# ── Clip recorder ─────────────────────────────────────────────────────────────

def record_night_vision_clip(
    get_frame,
    duration: int = 10,
    filename: str  = "night_clip.mp4",
    cam_id: int    = 0,
    nv_mode: str   = "green",
) -> str:
    """
    Records a night-vision clip.

    Workflow: write NV frames → temp .avi → ffmpeg H.264 → .mp4 → delete .avi
    Falls back to raw .avi when ffmpeg is absent.

    Returns the path of the finished file.
    """
    if not filename.lower().endswith(".mp4"):
        filename = os.path.splitext(filename)[0] + ".mp4"

    tmp_avi = filename.replace(".mp4", "_nv_tmp.avi")

    # ── Write NV frames ───────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out    = cv2.VideoWriter(tmp_avi, fourcc, 15.0, (640, 480))

    if not out.isOpened():
        print(f"[NV] ERROR: VideoWriter failed to open {tmp_avi}")
        return filename

    start, frames_written = time.time(), 0
    while time.time() - start < duration:
        frame = get_frame()
        if frame is not None:
            nv  = enhance_night_vision(frame, mode=nv_mode)
            nv  = cv2.resize(nv, (640, 480))
            ts  = time.strftime("%H:%M:%S")
            mode_label = nv_mode.upper()
            cv2.putText(nv, f"NV-{mode_label}  CAM{cam_id}  {ts}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.60, (0, 255, 0), 1)
            out.write(nv)
            frames_written += 1
        time.sleep(1 / 15)
    out.release()

    print(f"[NV] Wrote {frames_written} frames to {tmp_avi}")
    if frames_written == 0:
        return filename

    # ── Convert to H.264 MP4 ──────────────────────────────────────
    try:
        result = subprocess.run(
            ["ffmpeg", "-y",
             "-i",  tmp_avi,
             "-c:v", "libx264",
             "-preset", "fast",
             "-crf",  "26",
             "-pix_fmt", "yuv420p",
             filename],
            capture_output=True, timeout=60,
        )
        if result.returncode == 0 and os.path.exists(filename) \
                and os.path.getsize(filename) > 0:
            os.remove(tmp_avi)
            print(f"[NV] ✅ MP4 ready: {filename}")
            return filename
        else:
            err = result.stderr.decode(errors="replace")[-300:]
            print(f"[NV] ffmpeg failed (rc={result.returncode}): {err}")
            print(f"[NV] Falling back to AVI: {tmp_avi}")
            return tmp_avi
    except FileNotFoundError:
        print("[NV] ffmpeg not found — sending raw AVI.")
        return tmp_avi
    except subprocess.TimeoutExpired:
        print("[NV] ffmpeg timed out — sending raw AVI.")
        return tmp_avi
    except Exception as e:
        print(f"[NV] Unexpected: {e}")
        return tmp_avi


# ── Live NV preview window ────────────────────────────────────────────────────

def night_camera(get_frame, cam_id: int = 0, mode: str = "green"):
    print(f"Night Vision [{mode}] active — Camera {cam_id}  (press n / t / w to switch, q to close)")
    current_mode = mode
    while True:
        frame = get_frame()
        if frame is None:
            continue
        nv = enhance_night_vision(frame, mode=current_mode)
        cv2.putText(nv, f"NV [{current_mode.upper()}]  CAM {cam_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.imshow(f"Night Vision — Cam {cam_id}", nv)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("t"):
            current_mode = "thermal"
        elif key == ord("g"):
            current_mode = "green"
        elif key == ord("w"):
            current_mode = "white"
    cv2.destroyWindow(f"Night Vision — Cam {cam_id}")
