#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Enrico Santagati

"""Extract mel-spectrogram features from audio for micro_speech validation.

Uses test .wav files from the tflite-micro repo (or custom .wav files) to
produce int8 feature tensors matching the micro_speech model input format.

Usage:
    pip install -r requirements.txt
    python extract_mfcc.py                           # Use tflite-micro test data
    python extract_mfcc.py --wav-dir /path/to/wavs   # Use custom .wav files

Output:
    test_samples/<label>.bin  — 1960-byte raw int8 tensor (49 frames x 40 channels)
    test_samples/<label>.hex  — hex string for tflite_micro.write_input(data_hex=...)
"""

import argparse
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# micro_speech model parameters
# (from audio_preprocessor.py / micro_model_settings.h)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
WINDOW_SIZE_MS = 30
WINDOW_STRIDE_MS = 20
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SIZE_MS // 1000   # 480
STRIDE_SAMPLES = SAMPLE_RATE * WINDOW_STRIDE_MS // 1000  # 320
NUM_FRAMES = 49
NUM_CHANNELS = 40
FFT_SIZE = 512
MEL_LOWER_HZ = 125.0
MEL_UPPER_HZ = 7500.0
INPUT_SIZE = NUM_FRAMES * NUM_CHANNELS  # 1960 bytes

# Default: tflite-micro test data in the Nordic SDK
TFLM_TESTDATA = Path(
    "/opt/nordic/ncs/v3.2.2/optional/modules/lib/tflite-micro/"
    "tensorflow/lite/micro/examples/micro_speech/testdata"
)

# Test samples to process — (output label, wav filename)
DEFAULT_SAMPLES = [
    ("yes",     "yes_1000ms.wav"),
    ("no",      "no_1000ms.wav"),
    ("silence", "silence_1000ms.wav"),
    ("noise",   "noise_1000ms.wav"),
]

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "test_samples"


def load_wav(path: Path) -> np.ndarray:
    """Load a .wav file as int16 samples."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="int16")
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE}Hz, got {sr}Hz in {path.name}")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio


def compute_mel_features(audio_int16: np.ndarray) -> np.ndarray:
    """Compute log-mel spectrogram features matching micro_speech pipeline.

    The firmware pipeline operates on raw int16 PCM and does:
      1. Hann window (480 samples)
      2. 512-point FFT
      3. 40-channel mel filterbank (125-7500 Hz)
      4. Square root, noise reduction (spectral subtraction), PCAN
      5. Log2 scaling * 64 (post_scaling_bits=6) → int16 range ~0-670
      6. Quantize: int8 = (feature * 256 + 333) / 666 - 128

    We approximate steps 1-5 with librosa's mel spectrogram (amplitude) + log2,
    keeping int16 scale (no normalization to [-1,1]) so magnitudes match.
    Steps 4 (noise reduction, PCAN) are skipped — this is an approximation.

    Returns:
        int8 numpy array, shape (1960,) — flattened [49, 40] tensor
    """
    import librosa

    # Pad or trim to exactly 1 second
    target_len = SAMPLE_RATE
    if len(audio_int16) < target_len:
        audio_int16 = np.pad(audio_int16, (0, target_len - len(audio_int16)))
    else:
        audio_int16 = audio_int16[:target_len]

    # Keep int16 scale — do NOT normalize to [-1, 1].
    # The firmware processes raw int16 PCM, so energy magnitudes are large.
    audio_float = audio_int16.astype(np.float32)

    # Compute amplitude mel spectrogram (power=1.0 ≈ firmware's sqrt step)
    mel_spec = librosa.feature.melspectrogram(
        y=audio_float,
        sr=SAMPLE_RATE,
        n_fft=FFT_SIZE,
        hop_length=STRIDE_SAMPLES,
        win_length=WINDOW_SAMPLES,
        window="hann",
        n_mels=NUM_CHANNELS,
        fmin=MEL_LOWER_HZ,
        fmax=MEL_UPPER_HZ,
        power=1.0,
    )
    # shape: (40, T) — need exactly 49 frames
    if mel_spec.shape[1] < NUM_FRAMES:
        mel_spec = np.pad(mel_spec, ((0, 0), (0, NUM_FRAMES - mel_spec.shape[1])))
    mel_spec = mel_spec[:, :NUM_FRAMES]

    # Log2 scaling with post_scaling_bits=6: feature = log2(energy) * 64
    # Floor at 1.0 to avoid log2(0); silence → 0
    log_mel = np.log2(np.maximum(mel_spec, 1.0)) * 64.0

    # Firmware quantization formula (from audio_preprocessor.py lines 218-231):
    #   value_div = int(25.6 * 26.0 + 0.5) = 666
    #   value = (feature * 256 + 333) / 666 - 128
    #   clip to [-128, 127]
    value_div = int(25.6 * 26.0 + 0.5)  # 666
    features = ((log_mel * 256) + (value_div // 2)) // value_div - 128
    features = np.clip(features, -128, 127).astype(np.int8)

    # Transpose to [frames, channels] and flatten for model input [1, 49, 40]
    features = features.T.flatten()
    assert features.shape == (INPUT_SIZE,), f"Got {features.shape}, want ({INPUT_SIZE},)"
    return features


def save_features(features: np.ndarray, output_path: Path):
    """Save int8 features as .bin (raw) and .hex (text) files."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.tofile(output_path)

    hex_path = output_path.with_suffix(".hex")
    hex_path.write_text(features.tobytes().hex())

    print(f"  {output_path.name} ({len(features)} bytes)")
    print(f"  {hex_path.name} ({len(features) * 2} hex chars)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wav-dir", type=Path, default=None,
                        help="Directory with .wav files (default: tflite-micro testdata)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    wav_dir = args.wav_dir or TFLM_TESTDATA
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not wav_dir.exists():
        print(f"ERROR: WAV directory not found: {wav_dir}")
        print("Install the Nordic SDK or specify --wav-dir with your own .wav files.")
        return 1

    print(f"WAV source:  {wav_dir}")
    print(f"Output dir:  {out_dir}\n")

    if args.wav_dir is None:
        samples = [(label, wav_dir / fname) for label, fname in DEFAULT_SAMPLES]
    else:
        samples = [(p.stem, p) for p in sorted(wav_dir.glob("*.wav"))]

    if not samples:
        print("No .wav files found!")
        return 1

    for label, wav_path in samples:
        if not wav_path.exists():
            print(f"SKIP: {wav_path} not found")
            continue

        print(f"'{label}' <- {wav_path.name}")
        audio = load_wav(wav_path)
        features = compute_mel_features(audio)
        save_features(features, out_dir / f"{label}.bin")
        print(f"  min={features.min():4d}  max={features.max():3d}  "
              f"mean={features.mean():6.1f}  nonzero={np.count_nonzero(features)}")
        print()

    print("Injection workflow:")
    print("  tflite_micro.write_input(data_hex=open('test_samples/yes.hex').read())")
    print("  tflite_micro.read_output()")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
