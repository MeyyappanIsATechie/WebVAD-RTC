# vad/energy_vad.py
# Step 2 — Energy-based VAD (our baseline detector)
# Concept: chop audio into frames → compute RMS per frame → threshold → segments

import numpy as np
import matplotlib.pyplot as plt

# Import our preprocessing tools from Step 1
# (make sure preprocessing.py is in the same folder)
from preprocessing import generate_synthetic_audio, TARGET_SR


# ─────────────────────────────────────────────
# FRAMING
# ─────────────────────────────────────────────

def frame_audio(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    frame_ms: int = 20,       # frame length in milliseconds
    hop_ms: int = 10,         # hop size (how far we step between frames)
) -> np.ndarray:
    """
    Split a 1D audio array into overlapping frames.

    Args:
        audio     : 1D float32 array of audio samples
        sr        : sample rate (samples per second)
        frame_ms  : length of each frame in milliseconds (default: 20ms)
        hop_ms    : step between frame starts in ms (default: 10ms → 50% overlap)

    Returns:
        frames: 2D array of shape (num_frames, frame_length)

    Why overlap?
    With 0% overlap you might catch a speech onset right at a frame boundary
    and miss it. A 50% overlap (hop = half frame) ensures every sample is
    covered by at least 2 frames — much more robust.

    Analogy: scanning a newspaper with a flashlight.
    Moving it in big jumps means you might skip a line.
    Smaller, overlapping moves = nothing missed.
    """
    frame_len = int(sr * frame_ms / 1000)   # e.g. 20ms @ 16kHz = 320 samples
    hop_len   = int(sr * hop_ms  / 1000)    # e.g. 10ms @ 16kHz = 160 samples

    # How many complete frames fit?
    num_frames = 1 + (len(audio) - frame_len) // hop_len

    # Build 2D array: each row is one frame
    frames = np.stack([
        audio[i * hop_len : i * hop_len + frame_len]
        for i in range(num_frames)
    ])

    return frames   # shape: (num_frames, frame_len)


# ─────────────────────────────────────────────
# ENERGY COMPUTATION
# ─────────────────────────────────────────────

def compute_rms(frames: np.ndarray) -> np.ndarray:
    """
    Compute RMS energy for each frame.

    RMS = sqrt( mean( x^2 ) )

    Returns a 1D array of shape (num_frames,) — one energy value per frame.
    High value = loud frame (likely speech).
    Low value  = quiet frame (likely silence/noise).
    """
    return np.sqrt(np.mean(frames ** 2, axis=1))


# ─────────────────────────────────────────────
# THRESHOLDING
# ─────────────────────────────────────────────

def estimate_noise_floor(rms: np.ndarray, percentile: float = 20.0) -> float:
    """
    Estimate the background noise level from the RMS array.

    We take the Nth percentile of energy values — the assumption is
    that a significant portion of the audio IS silence/noise (not all speech).

    percentile=20 means: "the energy level that 20% of frames fall below."
    Those quiet frames represent the noise floor.

    Why not just take the minimum?
    The minimum can be a fluke (one dead-silent frame in a noisy room).
    A percentile is more robust.
    """
    return float(np.percentile(rms, percentile))


def compute_threshold(rms: np.ndarray, multiplier: float = 4.0) -> float:
    """
    Set the VAD decision threshold as a multiple of the estimated noise floor.

    threshold = noise_floor * multiplier

    multiplier=4.0 means: "if a frame is 4x louder than background noise,
    call it speech."

    This is the key tuning knob of energy VAD:
    - Too low → false positives (noise mistaken for speech)
    - Too high → false negatives (quiet speech missed)

    In practice, you'd tune this per-dataset or make it adaptive over time.
    """
    noise_floor = estimate_noise_floor(rms)
    threshold = noise_floor * multiplier
    return threshold


# ─────────────────────────────────────────────
# FRAME-LEVEL CLASSIFICATION
# ─────────────────────────────────────────────

def classify_frames(rms: np.ndarray, threshold: float) -> np.ndarray:
    """
    Binary classify each frame: 1 = SPEECH, 0 = SILENCE.

    This is the actual "decision" — everything before this was just
    computing features and a threshold. The decision itself is one line.
    """
    return (rms > threshold).astype(np.int8)


# ─────────────────────────────────────────────
# CONVERT FRAME LABELS → TIME SEGMENTS
# ─────────────────────────────────────────────

def frames_to_segments(
    labels: np.ndarray,
    sr: int = TARGET_SR,
    hop_ms: int = 10,
) -> list[tuple[float, float]]:
    """
    Convert a binary frame-label array into a list of (start_sec, end_sec) segments.

    Example:
      labels = [0, 0, 1, 1, 1, 0, 0, 1, 1, 0]
                         ↑─────↑          ↑──↑
      → [(segment1_start, segment1_end), (segment2_start, segment2_end)]

    This is useful for downstream systems (ASR, diarization) that need
    timestamps, not per-frame binary vectors.
    """
    hop_sec = hop_ms / 1000.0
    segments = []
    in_speech = False
    seg_start = 0.0

    for i, label in enumerate(labels):
        t = i * hop_sec   # time of this frame's start in seconds

        if label == 1 and not in_speech:
            # Transition: silence → speech
            seg_start = t
            in_speech = True

        elif label == 0 and in_speech:
            # Transition: speech → silence
            segments.append((seg_start, t))
            in_speech = False

    # If audio ends while still in speech, close the last segment
    if in_speech:
        segments.append((seg_start, len(labels) * hop_sec))

    return segments


# ─────────────────────────────────────────────
# EVALUATION (simple, frame-level)
# ─────────────────────────────────────────────

def evaluate(
    predicted_labels: np.ndarray,
    ground_truth_segments: list[tuple[float, float]],
    total_samples: int,
    sr: int = TARGET_SR,
    hop_ms: int = 10,
) -> dict:
    """
    Compare predicted frame labels against ground truth segments.

    Converts ground truth to a frame-level binary array, then computes:
    - Precision : of frames we called SPEECH, what fraction actually is?
    - Recall    : of actual SPEECH frames, what fraction did we catch?
    - F1        : harmonic mean of precision and recall

    Precision = TP / (TP + FP)   ← "how many of my 'speech' calls are right?"
    Recall    = TP / (TP + FN)   ← "how many real speech frames did I find?"
    F1        = 2 * P * R / (P + R)  ← single number balancing both

    Why F1 and not just accuracy?
    If 80% of your audio is silence, a dumb detector that always says
    "silence" gets 80% accuracy — but 0% recall. F1 punishes this.
    """
    hop_sec = hop_ms / 1000.0
    num_frames = len(predicted_labels)

    # Build ground-truth frame-level labels
    gt_labels = np.zeros(num_frames, dtype=np.int8)
    for (start_sec, end_sec) in ground_truth_segments:
        start_frame = int(start_sec / hop_sec)
        end_frame   = int(end_sec   / hop_sec)
        gt_labels[start_frame:end_frame] = 1

    TP = int(np.sum((predicted_labels == 1) & (gt_labels == 1)))
    FP = int(np.sum((predicted_labels == 1) & (gt_labels == 0)))
    FN = int(np.sum((predicted_labels == 0) & (gt_labels == 1)))
    TN = int(np.sum((predicted_labels == 0) & (gt_labels == 0)))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": round(precision, 3),
        "recall":    round(recall, 3),
        "f1":        round(f1, 3),
    }


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_vad_result(
    audio: np.ndarray,
    rms: np.ndarray,
    threshold: float,
    predicted_segments: list[tuple[float, float]],
    ground_truth: list[tuple[float, float]],
    sr: int = TARGET_SR,
    hop_ms: int = 10,
):
    """
    Three-panel plot:
      Top    : raw waveform with ground truth (green) and predictions (red) overlaid
      Middle : RMS energy curve with threshold line
      Bottom : frame-level binary decision (speech=1, silence=0)
    """
    hop_sec = hop_ms / 1000.0
    duration = len(audio) / sr
    time_audio = np.linspace(0, duration, len(audio))
    time_frames = np.arange(len(rms)) * hop_sec

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    fig.suptitle("Energy VAD — Results", fontsize=13, fontweight="bold")

    # ── Panel 1: Waveform ──
    ax = axes[0]
    ax.plot(time_audio, audio, color="#4A90D9", linewidth=0.5, label="Waveform")
    for i, (s, e) in enumerate(ground_truth):
        ax.axvspan(s, e, alpha=0.2, color="green",
                   label="Ground truth" if i == 0 else "")
    for i, (s, e) in enumerate(predicted_segments):
        ax.axvspan(s, e, alpha=0.2, color="red",
                   label="VAD prediction" if i == 0 else "")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Waveform (green=truth, red=prediction)")

    # ── Panel 2: RMS energy + threshold ──
    ax = axes[1]
    ax.plot(time_frames, rms, color="#E8A838", linewidth=1.2, label="RMS energy")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label=f"Threshold ({threshold:.4f})")
    ax.set_ylabel("RMS Energy")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Frame-level RMS Energy")

    # ── Panel 3: Binary decision ──
    labels = (rms > threshold).astype(int)
    ax = axes[2]
    ax.fill_between(time_frames, labels, step="post", color="#7B68EE", alpha=0.7, label="Speech=1")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Silence", "Speech"])
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Frame-level VAD Decision")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# MAIN — RUN THE FULL PIPELINE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── 1. Generate synthetic audio (from Step 1) ──
    print("Generating synthetic audio...")
    audio, gt_segments = generate_synthetic_audio(duration_sec=5.0)
    print(f"  Audio shape : {audio.shape}  ({len(audio)/TARGET_SR:.1f}s @ {TARGET_SR}Hz)")
    print(f"  Ground truth: {gt_segments}")

    # ── 2. Frame the audio ──
    frames = frame_audio(audio)
    print(f"\nFraming:")
    print(f"  Frame shape : {frames.shape}  ({frames.shape[0]} frames × {frames.shape[1]} samples)")

    # ── 3. Compute RMS energy per frame ──
    rms = compute_rms(frames)
    print(f"\nRMS Energy:")
    print(f"  Min  : {rms.min():.5f}")
    print(f"  Max  : {rms.max():.5f}")
    print(f"  Mean : {rms.mean():.5f}")

    # ── 4. Compute threshold & classify ──
    threshold = compute_threshold(rms, multiplier=4.0)
    labels    = classify_frames(rms, threshold)
    print(f"\nThreshold : {threshold:.5f}")
    print(f"Frames marked SPEECH  : {labels.sum()} / {len(labels)}")

    # ── 5. Convert to time segments ──
    pred_segments = frames_to_segments(labels)
    print(f"\nPredicted speech segments:")
    for s, e in pred_segments:
        print(f"  {s:.2f}s → {e:.2f}s")

    # ── 6. Evaluate ──
    metrics = evaluate(labels, gt_segments, total_samples=len(audio))
    print(f"\nEvaluation:")
    print(f"  Precision : {metrics['precision']}")
    print(f"  Recall    : {metrics['recall']}")
    print(f"  F1        : {metrics['f1']}")
    print(f"  TP={metrics['TP']}  FP={metrics['FP']}  FN={metrics['FN']}  TN={metrics['TN']}")

    # ── 7. Visualize ──
    plot_vad_result(audio, rms, threshold, pred_segments, gt_segments)