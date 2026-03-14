// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Enrico Santagati

/*
 * On-device audio preprocessing pipeline for micro_speech.
 *
 * Converts 1s of raw 16kHz int16 PCM → [49, 40] int8 MFCC features
 * using the tflite-micro signal library C functions.
 *
 * Pipeline per frame (30ms window, 20ms stride):
 *   1. Hann window (scaling_bits=12)
 *   2. FFT auto-scale
 *   3. 512-point RFFT (CMSIS-DSP arm_rfft_q15)
 *   4. Energy (|complex|²)
 *   5. 40-channel mel filterbank
 *   6. Square root + scale compensation
 *   7. Spectral subtraction (noise reduction)
 *   8. PCAN AGC
 *   9. Log scaling
 *  10. Int8 quantization
 */

#include "audio_frontend.h"

#include <string.h>
#include <zephyr/logging/log.h>

/* Signal library headers */
#include "signal/src/complex.h"
#include "signal/src/window.h"
#include "signal/src/fft_auto_scale.h"
#include "signal/src/energy.h"
#include "signal/src/filter_bank.h"
#include "signal/src/filter_bank_square_root.h"
#include "signal/src/filter_bank_spectral_subtraction.h"
#include "signal/src/pcan_argc_fixed.h"
#include "signal/src/filter_bank_log.h"

/* CMSIS-DSP FFT (replaces Kiss FFT for ~50x speedup) */
#include <arm_math.h>

LOG_MODULE_REGISTER(audio_frontend, LOG_LEVEL_INF);

/* ── Pipeline parameters ─────────────────────────────────────────────── */
#define SAMPLE_RATE             16000
#define WINDOW_SIZE_SAMPLES     480     /* 30 ms */
#define WINDOW_STRIDE_SAMPLES   320     /* 20 ms */
#define NUM_FRAMES              49
#define FFT_LENGTH              512
#define SPECTRUM_SIZE           (FFT_LENGTH / 2 + 1)  /* 257 */
#define NUM_CHANNELS            40
#define WINDOW_SCALING_BITS     12
#define FILTER_BANK_SCALING_BITS 12
#define POST_SCALING_BITS       6
#define SPECTRAL_SUB_BITS       14
#define SMOOTHING_BITS          10
#define FFT_SIZE_LOG2           9
#define INPUT_CORRECTION_BITS   (FFT_SIZE_LOG2 - FILTER_BANK_SCALING_BITS / 2)  /* 3 */
#define PCAN_SNR_SHIFT          6

/* Spectral subtraction config (computed from float params) */
#define EVEN_SMOOTHING          ((uint32_t)(0.025 * (1 << SPECTRAL_SUB_BITS)))   /* 409 */
#define ODD_SMOOTHING           ((uint32_t)(0.06  * (1 << SPECTRAL_SUB_BITS)))   /* 983 */
#define ONE_MINUS_EVEN          ((uint32_t)((1 << SPECTRAL_SUB_BITS) - EVEN_SMOOTHING))
#define ONE_MINUS_ODD           ((uint32_t)((1 << SPECTRAL_SUB_BITS) - ODD_SMOOTHING))
#define MIN_SIGNAL_REMAINING    ((uint32_t)(0.05  * (1 << SPECTRAL_SUB_BITS)))   /* 819 */

/* Int8 quantization constants */
#define VALUE_SCALE             256
#define VALUE_DIV               666     /* int((25.6 * 26.0) + 0.5) */

/* ── Extern const arrays from audio_frontend_data.cpp ────────────────── */
extern const int16_t kHannWindow[];
extern const int16_t kFilterbankWeights[];
extern const int16_t kFilterbankUnweights[];
extern const int16_t kChannelFrequencyStarts[];
extern const int16_t kChannelWeightStarts[];
extern const int16_t kChannelWidths[];
extern const int32_t kFilterbankStartIndex;
extern const int32_t kFilterbankEndIndex;
extern const int16_t kPcanGainLut[];

/* ── Static scratch buffers (not on stack — thread has only 4 KB) ──── */

/* CMSIS-DSP RFFT instance (initialized once) */
static arm_rfft_instance_q15 rfft_instance;

/* Per-frame scratch */
static int16_t windowed[FFT_LENGTH];                       /* zero-padded */
static int16_t scaled[FFT_LENGTH];
static int16_t fft_raw[FFT_LENGTH * 2];                    /* CMSIS RFFT output: interleaved re/im */
static Complex<int16_t> fft_output[SPECTRUM_SIZE];
static uint32_t energy[SPECTRUM_SIZE];
static uint64_t filterbank_accum[NUM_CHANNELS + 1];        /* +1 for scratch ch0 */
static uint32_t filterbank_sqrt[NUM_CHANNELS];
static uint32_t filterbank_noise_sub[NUM_CHANNELS];
static int16_t  log_output[NUM_CHANNELS];

/* Persistent state across frames */
static uint32_t noise_estimate[NUM_CHANNELS];

/* Filterbank config (filled once in init) */
static tflite::tflm_signal::FilterbankConfig fb_config;

/* Spectral subtraction config */
static tflite::tflm_signal::SpectralSubtractionConfig ss_config;

/* ── Init ────────────────────────────────────────────────────────────── */
int audio_frontend_init(void)
{
    /* Initialize CMSIS-DSP RFFT */
    arm_status status = arm_rfft_init_q15(&rfft_instance, FFT_LENGTH,
                                           0 /* forward */, 1 /* bit-reverse */);
    if (status != ARM_MATH_SUCCESS) {
        LOG_ERR("arm_rfft_init_q15 failed: %d", status);
        return -1;
    }
    LOG_INF("FFT: CMSIS-DSP arm_rfft_q15 (512-point)");

    /* Fill filterbank config */
    fb_config.num_channels = NUM_CHANNELS;
    fb_config.channel_frequency_starts = kChannelFrequencyStarts;
    fb_config.channel_weight_starts = kChannelWeightStarts;
    fb_config.channel_widths = kChannelWidths;
    fb_config.weights = kFilterbankWeights;
    fb_config.unweights = kFilterbankUnweights;
    fb_config.output_scale = 0;
    fb_config.input_correction_bits = INPUT_CORRECTION_BITS;

    /* Fill spectral subtraction config */
    ss_config.num_channels = NUM_CHANNELS;
    ss_config.smoothing = EVEN_SMOOTHING;
    ss_config.one_minus_smoothing = ONE_MINUS_EVEN;
    ss_config.min_signal_remaining = MIN_SIGNAL_REMAINING;
    ss_config.alternate_smoothing = ODD_SMOOTHING;
    ss_config.alternate_one_minus_smoothing = ONE_MINUS_ODD;
    ss_config.smoothing_bits = SMOOTHING_BITS;
    ss_config.spectral_subtraction_bits = SPECTRAL_SUB_BITS;
    ss_config.clamping = false;

    /* Clear persistent state */
    memset(noise_estimate, 0, sizeof(noise_estimate));

    LOG_INF("Audio frontend initialized (FFT=%d, channels=%d)",
            FFT_LENGTH, NUM_CHANNELS);
    return 0;
}

/* ── Reset ───────────────────────────────────────────────────────────── */
void audio_frontend_reset_state(void)
{
    memset(noise_estimate, 0, sizeof(noise_estimate));
}

/* ── Per-step profiling accumulators (cycles, not us — avoid rounding) ── */
static uint32_t prof_cycles[10];

/* ── Process one frame ───────────────────────────────────────────────── */
static void process_frame(const int16_t *frame, int8_t *out_features)
{
    uint32_t t0, t1;

    /* 1. Apply Hann window */
    t0 = k_cycle_get_32();
    tflm_signal::ApplyWindow(frame, kHannWindow, WINDOW_SIZE_SAMPLES,
                             WINDOW_SCALING_BITS, windowed);
    /* Zero-pad from WINDOW_SIZE_SAMPLES to FFT_LENGTH */
    memset(&windowed[WINDOW_SIZE_SAMPLES], 0,
           (FFT_LENGTH - WINDOW_SIZE_SAMPLES) * sizeof(int16_t));
    t1 = k_cycle_get_32();
    prof_cycles[0] += t1 - t0;

    /* 2. FFT auto-scale */
    t0 = k_cycle_get_32();
    int scaling_shift = tflite::tflm_signal::FftAutoScale(
        windowed, FFT_LENGTH, scaled);
    t1 = k_cycle_get_32();
    prof_cycles[1] += t1 - t0;

    /* 3. RFFT (CMSIS-DSP — ~50x faster than Kiss FFT) */
    t0 = k_cycle_get_32();
    arm_rfft_q15(&rfft_instance, scaled, fft_raw);
    /* Convert interleaved q15 output to Complex<int16_t> for downstream */
    for (int i = 0; i < SPECTRUM_SIZE; i++) {
        fft_output[i].real = fft_raw[2 * i];
        fft_output[i].imag = fft_raw[2 * i + 1];
    }
    t1 = k_cycle_get_32();
    prof_cycles[2] += t1 - t0;

    /* 4. Energy — compute only in the filterbank range, zero the rest */
    t0 = k_cycle_get_32();
    memset(energy, 0, sizeof(energy));
    tflite::tflm_signal::SpectrumToEnergy(
        fft_output, kFilterbankStartIndex, kFilterbankEndIndex, energy);
    t1 = k_cycle_get_32();
    prof_cycles[3] += t1 - t0;

    /* 5. Filterbank — accumulate into 40 channels */
    t0 = k_cycle_get_32();
    memset(filterbank_accum, 0, sizeof(filterbank_accum));
    tflite::tflm_signal::FilterbankAccumulateChannels(
        &fb_config, energy, filterbank_accum);
    t1 = k_cycle_get_32();
    prof_cycles[4] += t1 - t0;

    /* 6. Square root + scale compensation
     * CMSIS-DSP arm_rfft_q15 scales by N (9 bits), Kiss FFT scales by
     * 2N (10 bits). CMSIS output is 2x larger → energy 4x larger →
     * sqrt 2x larger. Add 1 to scale_down_bits to compensate. */
    t0 = k_cycle_get_32();
    tflite::tflm_signal::FilterbankSqrt(
        &filterbank_accum[1],  /* skip scratch element 0 */
        NUM_CHANNELS, scaling_shift + 1, filterbank_sqrt);
    t1 = k_cycle_get_32();
    prof_cycles[5] += t1 - t0;

    /* 7. Spectral subtraction (noise reduction) */
    t0 = k_cycle_get_32();
    tflite::tflm_signal::FilterbankSpectralSubtraction(
        &ss_config, filterbank_sqrt, filterbank_noise_sub, noise_estimate);
    t1 = k_cycle_get_32();
    prof_cycles[6] += t1 - t0;

    /* 8. PCAN AGC — modifies filterbank_noise_sub in-place */
    t0 = k_cycle_get_32();
    tflite::tflm_signal::ApplyPcanAutoGainControlFixed(
        kPcanGainLut, PCAN_SNR_SHIFT,
        noise_estimate, filterbank_noise_sub, NUM_CHANNELS);
    t1 = k_cycle_get_32();
    prof_cycles[7] += t1 - t0;

    /* 9. Log scaling */
    t0 = k_cycle_get_32();
    int32_t output_scale = 1 << POST_SCALING_BITS;  /* 64 */
    tflite::tflm_signal::FilterbankLog(
        filterbank_noise_sub, NUM_CHANNELS,
        output_scale, INPUT_CORRECTION_BITS, log_output);
    t1 = k_cycle_get_32();
    prof_cycles[8] += t1 - t0;

    /* 10. Quantize int16 → int8 */
    t0 = k_cycle_get_32();
    for (int i = 0; i < NUM_CHANNELS; i++) {
        int32_t value = ((int32_t)log_output[i] * VALUE_SCALE + VALUE_DIV / 2)
                        / VALUE_DIV;
        value -= 128;
        if (value < -128) value = -128;
        if (value > 127)  value = 127;
        out_features[i] = (int8_t)value;
    }
    t1 = k_cycle_get_32();
    prof_cycles[9] += t1 - t0;
}

/* ── Process full 1-second clip ──────────────────────────────────────── */
int audio_frontend_process(const int16_t *pcm_input, int8_t *output)
{
    /* Reset profiling accumulators */
    memset(prof_cycles, 0, sizeof(prof_cycles));

    uint32_t t0 = k_cycle_get_32();

    for (int frame = 0; frame < NUM_FRAMES; frame++) {
        const int16_t *frame_start = &pcm_input[frame * WINDOW_STRIDE_SAMPLES];
        int8_t *frame_out = &output[frame * NUM_CHANNELS];
        process_frame(frame_start, frame_out);
    }

    uint32_t t1 = k_cycle_get_32();
    uint32_t total_us = k_cyc_to_us_floor32(t1 - t0);
    LOG_INF("Preprocessing: %u us (%u us/frame)", total_us, total_us / NUM_FRAMES);

    static const char *step_names[] = {
        "window", "autoscale", "rfft", "energy", "filterbank",
        "sqrt", "specsub", "pcan", "log", "quantize"
    };
    for (int i = 0; i < 10; i++) {
        uint32_t us = k_cyc_to_us_floor32(prof_cycles[i]);
        LOG_INF("  %-10s %7u us (%u%%)", step_names[i], us,
                total_us ? (us * 100) / total_us : 0);
    }

    return 0;
}
