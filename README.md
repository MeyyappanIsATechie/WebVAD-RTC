# WebVAD-RTC

Voice Activity Detection pipeline built using WebRTC VAD with optional denoising and post-processing.

The project processes raw audio and outputs speech segments with timestamps. It is designed for low-latency inference and local execution without relying on external APIs.

## Problem

Given an audio file, classify each frame as either speech or non-speech, then combine consecutive speech frames into segments.

The implementation should:

- Handle different sample rates and channel configurations
- Work in noisy recordings
- Support streaming with minimal latency
- Produce stable speech boundaries

## Pipeline

```
Audio
  │
  ├── Preprocessing
  │     ├── Resample
  │     ├── Convert to mono
  │     └── Normalize
  │
  ├── Denoising (optional)
  │
  ├── Frame generation
  │
  ├── WebRTC VAD
  │
  ├── Post-processing
  │     ├── Hangover
  │     ├── Merge nearby segments
  │     └── Remove short segments
  │
  └── Speech segments
```

## Project Structure

```
.
├── data/
├── src/
│   ├── preprocessing/
│   ├── vad/
│   ├── denoise/
│   ├── postprocessing/
│   └── evaluation/
├── tests/
├── outputs/
├── requirements.txt
└── README.md
```

## Why WebRTC VAD?

Several approaches were considered:

| Approach | Notes |
|----------|------|
| Energy Threshold | Simple baseline but unreliable in noisy environments |
| Energy + ZCR | Better than energy alone, still noise sensitive |
| WebRTC VAD | Fast, lightweight, no training required |
| MFCC + Classifier | Requires labeled data |
| Deep Learning VAD | Better accuracy but unnecessary complexity for this project |

WebRTC VAD provides a good balance between accuracy, latency, and implementation effort.

## Edge Cases

- Leading/trailing silence
- Short pauses between words
- Low-SNR speech
- Stereo recordings
- Different sample rates
- Loud transient noises
- Streaming audio

## Evaluation

The output is evaluated using:

- Frame-level accuracy
- Precision / Recall / F1
- Segment boundaries
- Manual inspection of difficult samples

## Future Work

- Adaptive noise estimation
- Confidence scores
- Streaming interface
- Silero VAD comparison
- Benchmarking on labeled datasets

## References

- WebRTC VAD
- RNNoise
- DeepFilterNet
