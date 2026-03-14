// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Enrico Santagati

/*
 * Micro speech inference engine for TFLite Micro.
 * Runs keyword spotting (yes/no/silence/unknown) on synthetic audio input.
 */

#ifndef MICRO_SPEECH_H_
#define MICRO_SPEECH_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Category labels for the micro_speech model output */
#define CATEGORY_SILENCE  0
#define CATEGORY_UNKNOWN  1
#define CATEGORY_YES      2
#define CATEGORY_NO       3
#define CATEGORY_COUNT    4

/* Model input dimensions: 49 time frames x 40 MFCC features */
#define INPUT_TIME_FRAMES  49
#define INPUT_FEATURES     40
#define INPUT_SIZE         (INPUT_TIME_FRAMES * INPUT_FEATURES)

/* Inference result — scores are dequantized to 0.0..1.0 */
struct inference_result {
	float scores[CATEGORY_COUNT];
	int predicted_class;
	float confidence;
	uint32_t inference_time_us;
};

/* Initialize the TFLite Micro interpreter and allocate tensors */
int micro_speech_init(void);

/* Run inference on the current input buffer.
 * Returns 0 on success, negative on error. */
int micro_speech_infer(struct inference_result *result);

/* Get pointer to the input tensor buffer (INPUT_SIZE int8 values).
 * Use this to inject test data via debug probe. */
int8_t *micro_speech_get_input_buffer(void);

/* Get pointer to the output tensor buffer (CATEGORY_COUNT int8 values). */
int8_t *micro_speech_get_output_buffer(void);

/* Get the tensor arena pointer and size (for debug probe memory inspection) */
uint8_t *micro_speech_get_arena(uint32_t *size);

/* Fill input buffer with synthetic test data for a given category.
 * pattern: 0=silence, 1=noise, 2=yes-like, 3=no-like */
void micro_speech_generate_test_input(int pattern);

/* Get label string for a category index */
const char *micro_speech_get_label(int category);

#ifdef __cplusplus
}
#endif

#endif /* MICRO_SPEECH_H_ */
