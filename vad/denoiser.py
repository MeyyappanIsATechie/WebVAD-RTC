# vad/denoiser.py
# Step 4 — Spectral Subtraction Denoiser
#
# Why not RNNoise / DeepFilterNet?
#   - RNNoise is a C library — harder to integrate purely in Python
#   - DeepFilterNet needs PyTorch + pretrained weights (~50MB download)
#   - Spectral subtraction is implementable from scratch with scipy,
#     teaches the core concepts, and works without any model weights.
#   - In production you'd swap this for DeepFilterNet — articulate that
#     as an "improvement" if asked.
#
# No new pip installs needed — scipy is already in our stack.

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from preprocessing import generate_synthetic_audio, TARGET_SR
from energy_vad    import evaluate, frames_to_segments
from webrtc_vad    import run_webrtc_vad


# ─────────────────────────────────────────────
# STFT PARAMETERS
# ─────────────────────────────────────────────

N_FFT      = 512    # FFT window size = 32ms @ 16kHz
                    # Frequency resolution = sr / N_FFT = 31.25 Hz/bin
HOP_LENGTH = 128    # Step between windows = 8ms @ 16kHz
                    # 75% overlap → smooth, artifact-free reconstruction
WINDOW     = 'hann' # Hann window tapers edges to zero → avoids spectral leakage
                    # (without this, energy "bleeds" into neighbouring freq bins)


# ─────────────────────────────────────────────
# STFT / ISTFT WRAPPERS
# ─────────────────────────────────────────────

def stft(audio: np.ndarray, sr: int = TARGET_SR) -> tuple:
    """
    Short-Time Fourier Transform.
    Splits audio into overlapping frames, runs FFT on each.

    Returns:
        freqs   : frequency axis (Hz) — shape (N_FFT//2 + 1,)
        times   : time axis (s)       — shape (num_frames,)
        Zxx     : complex STFT matrix — shape (freq_bins, num_frames)
                  Each column = FFT of one frame
                  Each row    = one frequency bin over time
    """
    freqs, times, Zxx = sig.stft(
        audio,
        fs       = sr,
        window   = WINDOW,
        nperseg  = N_FFT,
        noverlap = N_FFT - HOP_LENGTH,
    )
    return freqs, times, Zxx


def istft(Zxx: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Inverse STFT — reconstructs audio from a (possibly modified) STFT matrix.

    Key insight: we only modify the MAGNITUDE, keep the original PHASE.
    Phase carries the fine timing structure of speech — if you destroy it,
    the audio sounds robotic. Keeping original phase = natural-sounding output.
    """
    _, audio_out = sig.istft(
        Zxx,
        fs       = sr,
        window   = WINDOW,
        nperseg  = N_FFT,
        noverlap = N_FFT - HOP_LENGTH,
    )
    return audio_out.astype(np.float32)


# ─────────────────────────────────────────────
# NOISE ESTIMATION
# ─────────────────────────────────────────────

def estimate_noise_spectrum(
    Zxx: np.ndarray,
    num_noise_frames: int = 20,
) -> np.ndarray:
    """
    Estimate the background noise spectrum from the first N frames.

    Assumption: the first ~160ms of audio (20 frames × 8ms) is noise-only.
    In practice this is common — recordings often start before the speaker begins.

    We take the MEAN magnitude across those frames → a stable noise profile.

    Returns shape: (freq_bins, 1)  ← the 1 is for broadcasting across time later.

    Improvement you can articulate:
      Instead of "first N frames", use a running min-statistics estimator
      that tracks the noise floor continuously, even during speech.
      This handles environments where noise level changes over time (e.g. wind gusts).
    """
    magnitude_noise = np.abs(Zxx[:, :num_noise_frames])  # (freq_bins, num_noise_frames)
    noise_spectrum  = np.mean(magnitude_noise, axis=1, keepdims=True)  # (freq_bins, 1)
    return noise_spectrum


# ─────────────────────────────────────────────
# CORE: SPECTRAL SUBTRACTION
# ─────────────────────────────────────────────

def spectral_subtraction(
    audio: np.ndarray,
    sr: int    = TARGET_SR,
    alpha: float = 2.0,   # over-subtraction factor (1.0 = exact subtraction)
    beta: float  = 0.01,  # spectral floor ratio (prevents "musical noise")
    num_noise_frames: int = 20,
) -> np.ndarray:
    """
    Denoise audio using spectral subtraction.

    The algorithm step by step:

    1. STFT(noisy_audio) → complex matrix Zxx
    2. magnitude = |Zxx|           ← how loud each freq bin is at each frame
    3. phase     = angle(Zxx)      ← the "timing" of each frequency component
    4. noise_est = mean(|Zxx[:, :N]|)  ← noise profile from quiet frames
    5. clean_mag = max(magnitude - α × noise_est, β × magnitude)
                   └── subtract noise ──┘  └── but don't go below floor ──┘
    6. clean_Zxx = clean_mag × e^(j × phase)   ← reattach original phase
    7. ISTFT(clean_Zxx) → denoised audio

    Args:
        alpha: higher = more aggressive noise removal
               1.0 = subtract exactly the noise estimate
               2.0 = subtract 2× the estimate (recommended — handles estimation error)
               >3.0 = starts producing audible artifacts
        beta:  0.0  = allow total nulling of frequency bins (produces musical noise)
               0.01 = keep 1% of original signal as minimum (recommended)
               0.1  = very conservative, leaves more noise but fewer artifacts

    Returns:
        denoised audio (float32, same length as input)
    """
    # ── 1 & 2 & 3: STFT → magnitude + phase ──
    _, _, Zxx      = stft(audio, sr)
    magnitude      = np.abs(Zxx)
    phase          = np.angle(Zxx)

    # ── 4: Estimate noise ──
    noise_spectrum = estimate_noise_spectrum(Zxx, num_noise_frames)

    # ── 5: Subtract with floor ──
    # Shape broadcasting: magnitude is (freq_bins, frames), noise_spectrum is (freq_bins, 1)
    # → subtraction broadcasts across all frames automatically
    subtracted     = magnitude - alpha * noise_spectrum
    floor          = beta * magnitude
    magnitude_clean = np.maximum(subtracted, floor)

    # ── 6: Reconstruct complex STFT with original phase ──
    # e^(j×phase) = cos(phase) + j×sin(phase) — Euler's formula
    # This is just "put the magnitude back onto the phase direction"
    Zxx_clean = magnitude_clean * np.exp(1j * phase)

    # ── 7: ISTFT → back to time domain ──
    audio_clean = istft(Zxx_clean, sr)

    # Trim/pad to match input length (STFT boundary effects can shift length slightly)
    if len(audio_clean) > len(audio):
        audio_clean = audio_clean[:len(audio)]
    elif len(audio_clean) < len(audio):
        audio_clean = np.pad(audio_clean, (0, len(audio) - len(audio_clean)))

    return audio_clean


# ─────────────────────────────────────────────
# FULL DENOISER PIPELINE (entry point)
# ─────────────────────────────────────────────

def denoise(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Public API for the denoiser — call this from the rest of the pipeline.
    Returns denoised float32 audio of the same length.
    """
    return spectral_subtraction(audio, sr, alpha=2.0, beta=0.01)


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def plot_spectrogram(ax, audio, sr, title):
    """Helper: plot a spectrogram on a given matplotlib axis."""
    _, _, Zxx = stft(audio, sr)
    magnitude_db = 20 * np.log10(np.abs(Zxx) + 1e-10)  # convert to dB

    # Only show 0–4kHz (most speech content lives here)
    freq_bins = Zxx.shape[0]
    max_freq_bin = int(4000 / (sr / 2) * freq_bins)

    ax.imshow(
        magnitude_db[:max_freq_bin, :],
        aspect='auto',
        origin='lower',
        cmap='magma',
        vmin=-60, vmax=0,   # dB range: -60dB (near silence) to 0dB (max)
        extent=[0, len(audio)/sr, 0, 4000],
    )
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)


def plot_denoiser_results(
    audio_noisy:    np.ndarray,
    audio_denoised: np.ndarray,
    gt_segments:    list[tuple[float, float]],
    sr: int = TARGET_SR,
):
    """
    5-panel plot showing before vs after denoising + spectrograms.

    Panel 1: Noisy waveform
    Panel 2: Denoised waveform
    Panel 3: Noisy spectrogram   ← bright = energy present at that freq/time
    Panel 4: Denoised spectrogram← noise floor visibly lifted
    Panel 5: Difference waveform ← what the denoiser actually removed
    """
    duration = len(audio_noisy) / sr
    t = np.linspace(0, duration, len(audio_noisy))

    fig, axes = plt.subplots(5, 1, figsize=(13, 13))
    fig.suptitle("Spectral Subtraction Denoiser — Before vs After", fontsize=13, fontweight="bold")

    # ── Waveforms ──
    for ax, audio, title, color in [
        (axes[0], audio_noisy,    "Noisy Waveform",    "#E84040"),
        (axes[1], audio_denoised, "Denoised Waveform", "#4CAF50"),
    ]:
        ax.plot(t, audio, color=color, linewidth=0.5)
        for i, (s, e) in enumerate(gt_segments):
            ax.axvspan(s, e, alpha=0.15, color="blue",
                       label="Speech region" if i == 0 else "")
        ax.set_ylabel("Amplitude")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)

    # ── Spectrograms ──
    plot_spectrogram(axes[2], audio_noisy,    sr, "Spectrogram — Noisy (noise floor visible across all freqs)")
    plot_spectrogram(axes[3], audio_denoised, sr, "Spectrogram — Denoised (noise floor suppressed, speech bands remain)")

    # ── Difference ──
    diff = audio_noisy - audio_denoised[:len(audio_noisy)]
    axes[4].plot(t, diff, color="#888888", linewidth=0.5)
    axes[4].set_ylabel("Amplitude")
    axes[4].set_xlabel("Time (seconds)")
    axes[4].set_title("Removed Signal (Noisy − Denoised) — ideally this is just noise")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# MAIN — FULL PIPELINE COMPARISON
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Generate noisy synthetic audio ──
    print("Generating noisy synthetic audio...")
    audio_clean, gt = generate_synthetic_audio(duration_sec=5.0)
    noise = np.random.randn(len(audio_clean)).astype(np.float32) * 0.20
    audio_noisy = np.clip(audio_clean + noise, -1.0, 1.0)

    # ── Denoise ──
    print("Running spectral subtraction denoiser...")
    audio_denoised = denoise(audio_noisy)

    # ── Visualize before/after ──
    plot_denoiser_results(audio_noisy, audio_denoised, gt)

    # ─────────────────────────────────────────
    # Pipeline A: Noisy → WebRTC (no denoiser)
    # ─────────────────────────────────────────
    labels_a, segs_a = run_webrtc_vad(audio_noisy, aggressiveness=2)
    m_a = evaluate(labels_a, gt, total_samples=len(audio_noisy), hop_ms=20)

    # ─────────────────────────────────────────
    # Pipeline B: Noisy → Denoiser → WebRTC
    # ─────────────────────────────────────────
    labels_b, segs_b = run_webrtc_vad(audio_denoised, aggressiveness=2)
    m_b = evaluate(labels_b, gt, total_samples=len(audio_denoised), hop_ms=20)

    # ── Print comparison ──
    print(f"\n{'═'*52}")
    print(f"  Pipeline Comparison (noisy audio, mode=2)")
    print(f"{'═'*52}")
    print(f"  {'Pipeline':<30} │ Prec  │ Rec   │  F1")
    print(f"  {'─'*30}─┼───────┼───────┼──────")
    print(f"  {'Noisy → WebRTC':<30} │ {m_a['precision']:.3f} │ {m_a['recall']:.3f} │ {m_a['f1']:.3f}")
    print(f"  {'Noisy → Denoiser → WebRTC':<30} │ {m_b['precision']:.3f} │ {m_b['recall']:.3f} │ {m_b['f1']:.3f}")
    print(f"{'═'*52}")

    print(f"\nDenoised speech segments:")
    for s, e in segs_b:
        print(f"  {s:.2f}s → {e:.2f}s")