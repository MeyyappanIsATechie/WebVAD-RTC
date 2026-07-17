# vad/webrtc_vad.py
# Step 3 — WebRTC VAD integration
# Wraps Google's WebRTC VAD (GMM-based) into our pipeline.
# Also runs a side-by-side comparison vs energy VAD on noisy audio.

import numpy as np
import matplotlib.pyplot as plt
import webrtcvad

from preprocessing import generate_synthetic_audio, TARGET_SR
from energy_vad import (
    frame_audio, compute_rms, compute_threshold,
    classify_frames, frames_to_segments, evaluate
)


# ─────────────────────────────────────────────
# AUDIO FORMAT CONVERSION
# ─────────────────────────────────────────────

def float32_to_int16_bytes(audio: np.ndarray) -> bytes:
    """
    Convert our float32 audio array → raw int16 bytes.

    Why?
    WebRTC VAD is a C library under the hood. It expects audio in the format
    audio hardware naturally produces: 16-bit signed integers (PCM format).
    Our pipeline works in float32 for math convenience, so we convert here.

    float32 range : -1.0 to +1.0
    int16 range   : -32768 to +32767  (= -2^15 to 2^15 - 1)

    So we multiply by 32767 and cast.
    We clip first to prevent any floating point overshoot from causing overflow.
    """
    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int16   = (audio_clipped * 32767).astype(np.int16)
    return audio_int16.tobytes()   # raw bytes, no headers


# ─────────────────────────────────────────────
# CORE: WEBRTC VAD RUNNER
# ─────────────────────────────────────────────

def run_webrtc_vad(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    frame_ms: int = 20,          # must be 10, 20, or 30 — hard WebRTC constraint
    aggressiveness: int = 2,     # 0 (lenient) → 3 (strict)
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """
    Run WebRTC VAD on audio and return frame-level labels + speech segments.

    How WebRTC VAD works per frame:
      1. Takes exactly frame_ms milliseconds of int16 audio
      2. Computes spectral features (sub-band energies across frequency bands)
      3. Evaluates those features against its pre-trained GMM
      4. Returns True (speech) or False (non-speech)

    Args:
        audio         : float32 mono audio array (preprocessed)
        sr            : sample rate — must be 8000, 16000, 32000, or 48000
        frame_ms      : frame length in ms — must be 10, 20, or 30
        aggressiveness: 0=most permissive, 3=most strict

    Returns:
        labels   : int8 array of shape (num_frames,) — 1=speech, 0=silence
        segments : list of (start_sec, end_sec) tuples
    """
    assert frame_ms in (10, 20, 30), "WebRTC VAD only accepts 10, 20, or 30ms frames."
    assert sr in (8000, 16000, 32000, 48000), "WebRTC VAD only accepts 8/16/32/48kHz."

    vad = webrtcvad.Vad(aggressiveness)

    frame_len   = int(sr * frame_ms / 1000)   # samples per frame
    hop_len     = frame_len                    # no overlap for WebRTC (it handles it internally)
    frame_bytes = frame_len * 2               # int16 = 2 bytes per sample

    audio_bytes = float32_to_int16_bytes(audio)
    num_frames  = len(audio) // frame_len

    labels = np.zeros(num_frames, dtype=np.int8)

    for i in range(num_frames):
        start = i * frame_bytes
        end   = start + frame_bytes
        chunk = audio_bytes[start:end]

        if len(chunk) < frame_bytes:
            break   # skip incomplete last frame

        # This is the actual WebRTC GMM decision — one call per frame
        is_speech  = vad.is_speech(chunk, sr)
        labels[i]  = 1 if is_speech else 0

    segments = frames_to_segments(labels, sr=sr, hop_ms=frame_ms)
    return labels, segments


# ─────────────────────────────────────────────
# AGGRESSIVENESS SWEEP
# ─────────────────────────────────────────────

def sweep_aggressiveness(
    audio: np.ndarray,
    gt_segments: list[tuple[float, float]],
    sr: int = TARGET_SR,
    frame_ms: int = 20,
) -> None:
    """
    Run WebRTC VAD at all 4 aggressiveness levels and print evaluation metrics.

    This helps you pick the right aggressiveness for a given noise environment:
    - Low noise, clean audio  → aggressiveness 0 or 1
    - Moderate noise          → aggressiveness 2  (good default)
    - High noise environment  → aggressiveness 3
    """
    print(f"\n{'─'*55}")
    print(f"  Aggressiveness Sweep (frame={frame_ms}ms, sr={sr}Hz)")
    print(f"{'─'*55}")
    print(f"  Mode │ Precision │ Recall │   F1   │ Speech frames")
    print(f"{'─'*55}")

    for mode in range(4):
        labels, segments = run_webrtc_vad(audio, sr, frame_ms, aggressiveness=mode)
        m = evaluate(labels, gt_segments, total_samples=len(audio),
                     sr=sr, hop_ms=frame_ms)
        speech_count = labels.sum()
        print(f"   {mode}   │   {m['precision']:.3f}   │  {m['recall']:.3f} │ {m['f1']:.3f}  │ {speech_count}")

    print(f"{'─'*55}\n")


# ─────────────────────────────────────────────
# COMPARISON PLOT: Energy VAD vs WebRTC VAD
# ─────────────────────────────────────────────

def plot_comparison(
    audio: np.ndarray,
    gt_segments: list[tuple[float, float]],
    energy_labels: np.ndarray,
    webrtc_labels: np.ndarray,
    sr: int = TARGET_SR,
    frame_ms: int = 20,
    hop_ms_energy: int = 10,
    noise_level: float = 0.0,
):
    """
    4-panel plot comparing Energy VAD vs WebRTC VAD on the same audio.

    Panel 1: Raw waveform + ground truth
    Panel 2: Energy VAD binary decision
    Panel 3: WebRTC VAD binary decision
    Panel 4: Disagreement map (where the two methods differ)
    """
    hop_sec_e = hop_ms_energy / 1000.0
    hop_sec_w = frame_ms / 1000.0
    duration  = len(audio) / sr

    t_audio   = np.linspace(0, duration, len(audio))
    t_energy  = np.arange(len(energy_labels)) * hop_sec_e
    t_webrtc  = np.arange(len(webrtc_labels)) * hop_sec_w

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    fig.suptitle(f"Energy VAD vs WebRTC VAD  (noise level={noise_level})", fontsize=12, fontweight="bold")

    # ── Panel 1: Waveform + ground truth ──
    ax = axes[0]
    ax.plot(t_audio, audio, color="#4A90D9", linewidth=0.5)
    for i, (s, e) in enumerate(gt_segments):
        ax.axvspan(s, e, alpha=0.25, color="green",
                   label="Ground truth" if i == 0 else "")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform (green = ground truth speech)")
    ax.legend(loc="upper right", fontsize=8)

    # ── Panel 2: Energy VAD ──
    ax = axes[1]
    ax.fill_between(t_energy, energy_labels, step="post",
                    color="#E8A838", alpha=0.8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Silence", "Speech"])
    ax.set_title("Energy VAD Decision")

    # ── Panel 3: WebRTC VAD ──
    ax = axes[2]
    ax.fill_between(t_webrtc, webrtc_labels, step="post",
                    color="#7B68EE", alpha=0.8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Silence", "Speech"])
    ax.set_title("WebRTC VAD Decision (mode=2)")

    # ── Panel 4: Disagreement ──
    # Align both label arrays to the same timeline for comparison
    # (energy has 10ms hops, webrtc has 20ms hops → upsample webrtc)
    webrtc_upsampled = np.repeat(webrtc_labels, 2)[:len(energy_labels)]
    disagree = np.abs(energy_labels.astype(int) - webrtc_upsampled.astype(int))

    ax = axes[3]
    ax.fill_between(t_energy[:len(disagree)], disagree, step="post",
                    color="#E84040", alpha=0.7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Agree", "Disagree"])
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Disagreement Map (where they differ)")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Test on CLEAN audio first ──
    print("=" * 55)
    print("  TEST 1: Clean audio (low noise)")
    print("=" * 55)
    audio_clean, gt = generate_synthetic_audio(duration_sec=5.0)

    # Energy VAD on clean audio
    frames_e     = frame_audio(audio_clean, hop_ms=10)
    rms          = compute_rms(frames_e)
    thresh       = compute_threshold(rms, multiplier=4.0)
    energy_labels = classify_frames(rms, thresh)
    energy_segs  = frames_to_segments(energy_labels)
    m_energy     = evaluate(energy_labels, gt, total_samples=len(audio_clean))
    print(f"\nEnergy VAD  → F1: {m_energy['f1']}  P: {m_energy['precision']}  R: {m_energy['recall']}")

    # WebRTC VAD on clean audio (aggressiveness=2)
    webrtc_labels, webrtc_segs = run_webrtc_vad(audio_clean, aggressiveness=2)
    m_webrtc = evaluate(webrtc_labels, gt, total_samples=len(audio_clean), hop_ms=20)
    print(f"WebRTC VAD  → F1: {m_webrtc['f1']}  P: {m_webrtc['precision']}  R: {m_webrtc['recall']}")

    plot_comparison(audio_clean, gt, energy_labels, webrtc_labels, noise_level=0.04)

    # ── Aggressiveness sweep on clean audio ──
    sweep_aggressiveness(audio_clean, gt)

    # ── Test on NOISY audio ──
    print("=" * 55)
    print("  TEST 2: Noisy audio (high noise)")
    print("=" * 55)
    audio_noisy, gt = generate_synthetic_audio(
        duration_sec=5.0,
        speech_segments=[(0.5, 1.5), (2.0, 3.0), (3.5, 4.5)]
    )
    # Manually bump noise level by adding more noise
    noise_level = 0.20
    audio_noisy = audio_noisy + np.random.randn(len(audio_noisy)).astype(np.float32) * noise_level
    audio_noisy = np.clip(audio_noisy, -1.0, 1.0)

    # Energy VAD on noisy audio
    frames_e      = frame_audio(audio_noisy, hop_ms=10)
    rms           = compute_rms(frames_e)
    thresh        = compute_threshold(rms, multiplier=4.0)
    energy_labels = classify_frames(rms, thresh)
    m_energy      = evaluate(energy_labels, gt, total_samples=len(audio_noisy))
    print(f"\nEnergy VAD  → F1: {m_energy['f1']}  P: {m_energy['precision']}  R: {m_energy['recall']}")

    # WebRTC VAD on noisy audio
    webrtc_labels, _ = run_webrtc_vad(audio_noisy, aggressiveness=2)
    m_webrtc = evaluate(webrtc_labels, gt, total_samples=len(audio_noisy), hop_ms=20)
    print(f"WebRTC VAD  → F1: {m_webrtc['f1']}  P: {m_webrtc['precision']}  R: {m_webrtc['recall']}")

    plot_comparison(audio_noisy, gt, energy_labels, webrtc_labels, noise_level=noise_level)

    # ── Aggressiveness sweep on noisy audio ──
    sweep_aggressiveness(audio_noisy, gt)