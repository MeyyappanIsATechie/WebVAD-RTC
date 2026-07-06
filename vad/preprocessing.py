# vad/preprocessing.py
# Step 1 of VAD pipeline — loads, resamples, mono-ifies, normalizes audio.
# Also includes a synthetic audio generator for testing without real files.

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

TARGET_SR = 16000   # 16kHz — standard for speech processing
TARGET_DTYPE = np.float32  # work in float32 throughout


# ─────────────────────────────────────────────
# CORE PREPROCESSING FUNCTIONS
# ─────────────────────────────────────────────

def load_audio(filepath: str) -> tuple[np.ndarray, int]:
    """
    Load a .wav file from disk.
    Returns (samples, sample_rate).

    samples: numpy array of shape (num_samples,) or (num_channels, num_samples)
    """
    sr, data = wav.read(filepath)

    # wav.read returns int16 by default. Convert to float32 in [-1.0, 1.0].
    # Why? So all our math (energy, thresholds) is scale-independent.
    if data.dtype == np.int16:
        data = data.astype(TARGET_DTYPE) / 32768.0   # 2^15 = 32768
    elif data.dtype == np.int32:
        data = data.astype(TARGET_DTYPE) / 2147483648.0 #2^31
    elif data.dtype != TARGET_DTYPE:
        data = data.astype(TARGET_DTYPE)

    return data, sr


def to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert stereo (or multi-channel) audio to mono by averaging channels.

    Stereo wav files are stored as shape (num_samples, 2) by scipy.
    We average the two channels — simple but effective.
    """
    if audio.ndim == 2:
        # Shape is (num_samples, num_channels) — average across channels
        audio = np.mean(audio, axis=1)
    return audio


def resample(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """
    Resample audio from orig_sr to target_sr.

    Why resample? WebRTC VAD (next step) expects 8/16/32/48kHz specifically.
    Our models and feature extractors will all assume 16kHz.

    Uses scipy's polyphase resampling — clean and accurate.
    """
    if orig_sr == target_sr:
        return audio  # nothing to do

    # Compute the rational fraction: new_sr / old_sr
    # e.g., 44100 → 16000 means resample by factor 16000/44100 ≈ 0.363
    num_samples_new = int(len(audio) * target_sr / orig_sr)
    resampled = signal.resample(audio, num_samples_new)
    return resampled.astype(TARGET_DTYPE)


def normalize(audio: np.ndarray, method: str = "peak") -> np.ndarray:
    """
    Normalize audio amplitude.

    method="peak"  → scale so max absolute value = 1.0 (prevents clipping)
    method="rms"   → scale so RMS energy = 0.1 (useful for energy thresholding)

    Why normalize? A file recorded close-mic vs far-mic can differ by 30dB.
    Without normalization, the same threshold won't work on both.
    """
    if method == "peak":
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
    elif method == "rms":
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            audio = audio * (0.1 / rms)
    return audio


def preprocess(filepath: str) -> tuple[np.ndarray, int]:
    """
    Full preprocessing pipeline for a real audio file.
    Load → mono → resample → normalize.
    Returns (processed_audio, sample_rate).
    """
    audio, sr = load_audio(filepath)
    audio = to_mono(audio)
    audio = resample(audio, sr)
    audio = normalize(audio)
    return audio, TARGET_SR


# ─────────────────────────────────────────────
# SYNTHETIC AUDIO GENERATOR
# (for testing without real audio files)
# ─────────────────────────────────────────────

def generate_synthetic_audio(
    duration_sec: float = 5.0,
    sr: int = TARGET_SR,
    speech_segments: list[tuple[float, float]] | None = None
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """
    Generate a synthetic audio signal with:
      - Low-level background noise (always present)
      - Sine-wave "speech" bursts at specified time ranges

    Returns (audio_array, ground_truth_segments)
    ground_truth_segments: list of (start_sec, end_sec) where "speech" is present
    — we'll use this later to evaluate our VAD's accuracy.

    Args:
        duration_sec: total clip length in seconds
        sr: sample rate
        speech_segments: list of (start_sec, end_sec) tuples.
                         If None, uses a default pattern.
    """
    total_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)

    # --- Background noise (always on, low amplitude) ---
    # np.random.randn generates Gaussian (white) noise
    noise_level = 0.04
    audio = np.random.randn(total_samples).astype(TARGET_DTYPE) * noise_level

    # --- Default speech pattern if none provided ---
    if speech_segments is None:
        speech_segments = [
            (0.5, 1.5),   # word 1
            (2.0, 3.0),   # word 2
            (3.5, 4.5),   # word 3
        ]

    # --- Add "speech" (sine wave burst) for each segment ---
    # A pure sine is not real speech, but it mimics a voiced sound's structure:
    # periodic, higher energy than noise, sits in the speech frequency range.
    speech_freq = 200    # Hz — rough pitch of human voice fundamental
    speech_amplitude = 0.4

    for (start_sec, end_sec) in speech_segments:
        start_idx = int(start_sec * sr)
        end_idx   = int(end_sec * sr)

        # Fade in/out to avoid clicks at segment edges (like natural speech onsets)
        segment_len = end_idx - start_idx
        fade_samples = int(0.02 * sr)   # 20ms fade

        sine = np.sin(2 * np.pi * speech_freq * t[start_idx:end_idx])

        # Apply fade envelope
        envelope = np.ones(segment_len)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

        audio[start_idx:end_idx] += (sine * envelope * speech_amplitude).astype(TARGET_DTYPE)

    return audio, speech_segments


# ─────────────────────────────────────────────
# VISUALIZATION HELPER
# ─────────────────────────────────────────────

def plot_waveform(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    title: str = "Waveform",
    ground_truth: list[tuple[float, float]] | None = None
):
    """
    Plot the audio waveform over time.
    Optionally overlay ground-truth speech segments as green shaded regions.
    We'll later extend this to overlay our VAD predictions as well.
    """
    duration = len(audio) / sr
    time_axis = np.linspace(0, duration, len(audio))

    plt.figure(figsize=(12, 3))
    plt.plot(time_axis, audio, color="#4A90D9", linewidth=0.6, label="Audio")

    if ground_truth:
        for i, (start, end) in enumerate(ground_truth):
            plt.axvspan(start, end, alpha=0.25, color="green",
                        label="Speech (ground truth)" if i == 0 else "")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# QUICK SANITY CHECK
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating synthetic audio...")
    audio, gt_segments = generate_synthetic_audio(duration_sec=5.0)

    print(f"  Shape     : {audio.shape}")             # (80000,) for 5s @ 16kHz
    print(f"  Sample rate: {TARGET_SR} Hz")
    print(f"  Duration  : {len(audio)/TARGET_SR:.2f}s")
    print(f"  Min/Max   : {audio.min():.3f} / {audio.max():.3f}")
    print(f"  Ground truth speech segments: {gt_segments}")

    plot_waveform(audio, title="Synthetic Audio — Speech + Noise", ground_truth=gt_segments)

    # Optionally save to disk to test load_audio() in the next step
    # wav.write("test_audio.wav", TARGET_SR, audio)
    # print("Saved test_audio.wav")