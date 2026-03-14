// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Enrico Santagati

/*
 * On-device audio preprocessing for micro_speech model.
 *
 * Converts 1 second of raw int16 PCM (16 kHz) into [49, 40] int8 MFCC
 * features using the tflite-micro signal library (same C code used during
 * model training).
 */

#ifndef AUDIO_FRONTEND_H_
#define AUDIO_FRONTEND_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize the audio frontend (FFT state, scratch buffers).
 * Must be called once before audio_frontend_process().
 * Returns 0 on success, negative on error.
 */
int audio_frontend_init(void);

/**
 * Process 1 second of 16 kHz int16 PCM into model-ready int8 features.
 *
 * @param pcm_input  16000 int16_t samples (1 second at 16 kHz)
 * @param output     1960 int8_t values (49 frames x 40 features)
 * @return 0 on success, negative on error
 */
int audio_frontend_process(const int16_t *pcm_input, int8_t *output);

/**
 * Reset stateful noise estimator (call before processing a new clip).
 */
void audio_frontend_reset_state(void);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_FRONTEND_H_ */
