#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Enrico Santagati

"""Generate audio_frontend_data.cpp with precomputed const arrays.

Produces:
  - Hann window weights int16_t[480]  (scaling_bits=12)
  - Mel filterbank config arrays      (40 channels, 125–7500 Hz, 512-pt FFT)
  - PCAN gain LUT int16_t[125]        (strength=0.95, offset=80.0, gain_bits=21)

All values are bit-exact with the tflite-micro signal library Python code.
"""

import math
import os
import struct
import sys

# ── Parameters (from audio_preprocessor.py FeatureParams defaults) ──────────
SAMPLE_RATE = 16000
WINDOW_SIZE_MS = 30
WINDOW_SCALING_BITS = 12
FFT_LENGTH = 512
SPECTRUM_SIZE = FFT_LENGTH // 2 + 1  # 257
NUM_CHANNELS = 40
LOWER_BAND_LIMIT = 125.0
UPPER_BAND_LIMIT = 7500.0
FILTER_BANK_WEIGHT_SCALING_BITS = 12
FILTER_BANK_ALIGNMENT = 4
FILTER_BANK_CHANNEL_BLOCK_SIZE = 4

PCAN_STRENGTH = 0.95
PCAN_OFFSET = 80.0
PCAN_GAIN_BITS = 21
PCAN_SMOOTHING_BITS = 10
FFT_SIZE_LOG2 = 9  # log2(512)
INPUT_CORRECTION_BITS = FFT_SIZE_LOG2 - FILTER_BANK_WEIGHT_SCALING_BITS // 2  # 9-6=3
PCAN_INPUT_BITS = PCAN_SMOOTHING_BITS - INPUT_CORRECTION_BITS  # 10-3=7

WINDOW_SIZE_SAMPLES = int(WINDOW_SIZE_MS * SAMPLE_RATE / 1000)  # 480


# ── freq_to_mel: uses float32 for bit-exactness with C code ────────────────
def freq_to_mel(freq):
    """HTK mel scale, float32 precision."""
    # Use struct to force float32 intermediates like the C code does
    f = struct.unpack('f', struct.pack('f', freq))[0]
    return struct.unpack('f', struct.pack('f',
        1127.0 * math.log1p(f / 700.0)))[0]


def f32(val):
    """Force value to float32."""
    return struct.unpack('f', struct.pack('f', val))[0]


# ── Hann window ─────────────────────────────────────────────────────────────
def gen_hann_window(size, shift):
    """Generate Hann window weights as int16, matching window_op.py."""
    weights = []
    arg = 2.0 * math.pi / size
    for i in range(size):
        w = 0.5 - 0.5 * math.cos(arg * (i + 0.5))
        weights.append(int(round(w * (1 << shift))))
    return weights


# ── Mel filterbank ──────────────────────────────────────────────────────────
def calc_center_freq(num_channels, lower_freq_limit, upper_freq_limit):
    """Calculate center frequencies, matching filter_bank_ops.py."""
    mel_lower = f32(freq_to_mel(lower_freq_limit))
    mel_upper = f32(freq_to_mel(upper_freq_limit))
    mel_span = f32(mel_upper - mel_lower)
    mel_spacing = f32(mel_span / f32(float(num_channels)))
    centers = []
    for i in range(1, num_channels + 1):
        centers.append(f32(mel_lower + f32(mel_spacing * f32(float(i)))))
    return centers


def quantize_filterbank_weight(float_weight, scale_bits):
    weight = int(float_weight * (1 << scale_bits))
    unweight = int((1.0 - float_weight) * (1 << scale_bits))
    return weight, unweight


def init_filter_bank_weights():
    """Port of filter_bank_ops.py _init_filter_bank_weights()."""
    spectrum_size = SPECTRUM_SIZE
    sample_rate = SAMPLE_RATE
    alignment = FILTER_BANK_ALIGNMENT
    channel_block_size = FILTER_BANK_CHANNEL_BLOCK_SIZE
    num_channels = NUM_CHANNELS
    lower_band_limit = LOWER_BAND_LIMIT
    upper_band_limit = UPPER_BAND_LIMIT

    item_size = 2  # int16
    if alignment < item_size:
        index_alignment = 1
    else:
        index_alignment = alignment // item_size

    channel_frequency_starts = [0] * (num_channels + 1)
    channel_weight_starts = [0] * (num_channels + 1)
    channel_widths = [0] * (num_channels + 1)

    actual_channel_starts = [0] * (num_channels + 1)
    actual_channel_widths = [0] * (num_channels + 1)

    center_mel_freqs = calc_center_freq(num_channels + 1, lower_band_limit,
                                         upper_band_limit)

    hz_per_sbin = (sample_rate / 2) / (spectrum_size - 1)
    start_index = round(1 + (lower_band_limit / hz_per_sbin))

    chan_freq_index_start = start_index
    weight_index_start = 0
    needs_zeros = 0

    for chan in range(num_channels + 1):
        freq_index = chan_freq_index_start
        while freq_to_mel(freq_index * hz_per_sbin) <= center_mel_freqs[chan]:
            freq_index += 1

        width = freq_index - chan_freq_index_start
        actual_channel_starts[chan] = chan_freq_index_start
        actual_channel_widths[chan] = width

        if width == 0:
            channel_frequency_starts[chan] = 0
            channel_weight_starts[chan] = 0
            channel_widths[chan] = channel_block_size
            if needs_zeros == 0:
                needs_zeros = 1
                for j in range(chan):
                    channel_weight_starts[j] += channel_block_size
                weight_index_start += channel_block_size
        else:
            aligned_start = (chan_freq_index_start // index_alignment) * index_alignment
            aligned_width = chan_freq_index_start - aligned_start + width
            padded_width = ((aligned_width - 1) // channel_block_size + 1) * channel_block_size

            channel_frequency_starts[chan] = aligned_start
            channel_weight_starts[chan] = weight_index_start
            channel_widths[chan] = padded_width
            weight_index_start += padded_width
        chan_freq_index_start = freq_index

    num_weights = weight_index_start
    weights = [0] * num_weights
    unweights = [0] * num_weights

    end_index = 0
    mel_low = freq_to_mel(lower_band_limit)
    for chan in range(num_channels + 1):
        frequency = actual_channel_starts[chan]
        num_frequencies = actual_channel_widths[chan]
        frequency_offset = frequency - channel_frequency_starts[chan]
        weight_start = channel_weight_starts[chan]
        if chan == 0:
            denom_val = mel_low
        else:
            denom_val = center_mel_freqs[chan - 1]
        for j in range(num_frequencies):
            num = f32(center_mel_freqs[chan] -
                      freq_to_mel(frequency * hz_per_sbin))
            den = f32(center_mel_freqs[chan] - denom_val)
            weight = num / den
            weight_index = weight_start + frequency_offset + j
            w, uw = quantize_filterbank_weight(weight,
                                               FILTER_BANK_WEIGHT_SCALING_BITS)
            weights[weight_index] = w
            unweights[weight_index] = uw
            frequency += 1
        if frequency > end_index:
            end_index = frequency

    return (start_index, end_index, weights, unweights,
            channel_frequency_starts, channel_weight_starts, channel_widths)


# ── PCAN gain LUT ──────────────────────────────────────────────────────────
def pcan_gain_lookup(strength, offset, gain_bits, input_bits, x):
    """Port of PcanGainLookupFunction from wide_dynamic_func_lut_wrapper.cc."""
    x_as_float = float(x) / float(1 << input_bits)
    gain_as_float = float(1 << gain_bits) * math.pow(x_as_float + offset,
                                                       -strength)
    if gain_as_float > 32767.0:
        return 32767
    return int(gain_as_float + 0.5)


def gen_pcan_gain_lut(strength, offset, input_bits, gain_bits):
    """Port of WideDynamicFuncLut from wide_dynamic_func_lut_wrapper.cc."""
    BITS = 32
    LUT_SIZE = 4 * BITS - 3  # 125

    # Use a larger buffer with offset like the C code
    storage = [0] * (LUT_SIZE + 7)  # extra padding
    # In C: gain_lut points to storage[0], then -= 6 shifts it back
    # We'll use direct indexing with an offset

    # First two entries (interval 0 and 1)
    y0 = pcan_gain_lookup(strength, offset, gain_bits, input_bits, 0)
    y1 = pcan_gain_lookup(strength, offset, gain_bits, input_bits, 1)
    storage[0] = y0
    storage[1] = y1

    # For intervals 2..32, write at index [4*interval - 6] relative to storage
    for interval in range(2, BITS + 1):
        x0 = 1 << (interval - 1)
        x1 = x0 + (x0 >> 1)
        if interval == BITS:
            x2 = x0 + (x0 - 1)
        else:
            x2 = 2 * x0

        y0_v = pcan_gain_lookup(strength, offset, gain_bits, input_bits, x0)
        y1_v = pcan_gain_lookup(strength, offset, gain_bits, input_bits, x1)
        y2_v = pcan_gain_lookup(strength, offset, gain_bits, input_bits, x2)

        diff1 = y1_v - y0_v
        diff2 = y2_v - y0_v
        a1 = 4 * diff1 - diff2
        a2 = diff2 - a1

        idx = 4 * interval - 6  # C code: gain_lut -= 6, then [4*interval]
        storage[idx] = y0_v
        storage[idx + 1] = _clamp_i16(a1)
        storage[idx + 2] = _clamp_i16(a2)
        storage[idx + 3] = 0

    return storage[:LUT_SIZE]


def _clamp_i16(val):
    if val > 32767:
        return 32767
    if val < -32768:
        return -32768
    return val


# ── Code generation ─────────────────────────────────────────────────────────
def format_array(values, per_line=12, fmt="{}"):
    lines = []
    for i in range(0, len(values), per_line):
        chunk = values[i:i+per_line]
        lines.append("    " + ", ".join(fmt.format(v) for v in chunk) + ",")
    return "\n".join(lines)


def main():
    # Generate all data
    hann = gen_hann_window(WINDOW_SIZE_SAMPLES, WINDOW_SCALING_BITS)

    (start_index, end_index, weights, unweights,
     ch_freq_starts, ch_weight_starts, ch_widths) = init_filter_bank_weights()

    pcan_lut = gen_pcan_gain_lut(PCAN_STRENGTH, PCAN_OFFSET,
                                  PCAN_INPUT_BITS, PCAN_GAIN_BITS)

    # Verify PCAN LUT against known test values
    expected_first = [32636, 32633, 32630, -6, 0, 0, 32624, -12, 0, 0]
    if pcan_lut[:10] != expected_first:
        print(f"WARNING: PCAN LUT mismatch! Got {pcan_lut[:10]}", file=sys.stderr)
        print(f"         Expected {expected_first}", file=sys.stderr)
    else:
        print("PCAN LUT verified against test data ✓")

    print(f"Hann window: {len(hann)} samples")
    print(f"Filterbank: start_index={start_index}, end_index={end_index}")
    print(f"  weights: {len(weights)} values")
    print(f"  channels: {NUM_CHANNELS}")
    print(f"PCAN LUT: {len(pcan_lut)} entries")

    # Generate C++ file
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "src", "tflm", "audio_frontend_data.cpp")

    with open(out_path, "w") as f:
        f.write("""\
/*
 * Auto-generated by scripts/gen_frontend_data.py
 * DO NOT EDIT — regenerate with: python3 scripts/gen_frontend_data.py
 *
 * Audio preprocessing constants for micro_speech model.
 * Parameters: sample_rate=16000, window=30ms(480), FFT=512,
 *             40 mel channels [125-7500 Hz], PCAN(0.95, 80.0)
 */

#include <stdint.h>

""")
        # Note: 'extern' is needed because in C++, 'const' at file scope
        # has internal linkage by default. We need external linkage so
        # audio_frontend.cpp can reference these symbols.

        # Hann window
        f.write(f"/* Hann window: {WINDOW_SIZE_SAMPLES} samples, "
                f"scaling_bits={WINDOW_SCALING_BITS} */\n")
        f.write(f"extern const int16_t kHannWindow[{WINDOW_SIZE_SAMPLES}] = {{\n")
        f.write(format_array(hann))
        f.write("\n};\n\n")

        # Filterbank weights
        f.write(f"/* Mel filterbank weights: {len(weights)} values, "
                f"scaling_bits={FILTER_BANK_WEIGHT_SCALING_BITS} */\n")
        f.write(f"extern const int16_t kFilterbankWeights[{len(weights)}] = {{\n")
        f.write(format_array(weights))
        f.write("\n};\n\n")

        f.write(f"extern const int16_t kFilterbankUnweights[{len(unweights)}] = {{\n")
        f.write(format_array(unweights))
        f.write("\n};\n\n")

        # Channel config arrays
        f.write(f"extern const int16_t kChannelFrequencyStarts"
                f"[{len(ch_freq_starts)}] = {{\n")
        f.write(format_array(ch_freq_starts))
        f.write("\n};\n\n")

        f.write(f"extern const int16_t kChannelWeightStarts"
                f"[{len(ch_weight_starts)}] = {{\n")
        f.write(format_array(ch_weight_starts))
        f.write("\n};\n\n")

        f.write(f"extern const int16_t kChannelWidths[{len(ch_widths)}] = {{\n")
        f.write(format_array(ch_widths))
        f.write("\n};\n\n")

        # Filterbank indices
        f.write(f"extern const int32_t kFilterbankStartIndex = {start_index};\n")
        f.write(f"extern const int32_t kFilterbankEndIndex = {end_index};\n\n")

        # PCAN gain LUT
        f.write(f"/* PCAN gain LUT: strength={PCAN_STRENGTH}, "
                f"offset={PCAN_OFFSET}, gain_bits={PCAN_GAIN_BITS}, "
                f"input_bits={PCAN_INPUT_BITS} */\n")
        f.write(f"extern const int16_t kPcanGainLut[{len(pcan_lut)}] = {{\n")
        f.write(format_array(pcan_lut))
        f.write("\n};\n")

    print(f"\nGenerated: {out_path}")


if __name__ == "__main__":
    main()
