// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Enrico Santagati

/*
 * Micro speech inference engine — TFLite Micro keyword spotting.
 *
 * Model: micro_speech quantized (18.8KB)
 * Input:  [1, 49, 40] int8 — 49 time frames x 40 MFCC features
 * Output: [1, 4] int8 — silence, unknown, yes, no
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "micro_speech.h"
#include "model_data.h"

LOG_MODULE_REGISTER(micro_speech, LOG_LEVEL_INF);

/* Labels corresponding to model output indices */
static const char *category_labels[CATEGORY_COUNT] = {
	"silence", "unknown", "yes", "no"
};

/*
 * Tensor arena — sized for the micro_speech model.
 * The model needs ~10KB for tensors + scratch buffers.
 * We allocate 16KB to leave headroom for experimentation.
 *
 * NOTE: This is intentionally a global so the debug probe plugin
 * can find it via ELF symbols.
 */
constexpr int kTensorArenaSize = 7168;  /* arena_used_bytes() = 6948 + headroom */
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

/* TFLite Micro state */
static const tflite::Model *model = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *input_tensor = nullptr;
static TfLiteTensor *output_tensor = nullptr;

/*
 * Resolved buffer addresses — exported as globals so the debug probe
 * plugin can find them via ELF symbol lookup after init.
 */
int8_t *tflm_input_buffer = nullptr;
int8_t *tflm_output_buffer = nullptr;
int32_t tflm_input_size = 0;
int32_t tflm_output_size = 0;
float tflm_output_scale = 0.0f;
int32_t tflm_output_zero_point = 0;
int32_t tflm_arena_used = 0;

extern "C" int micro_speech_init(void)
{
	tflite::InitializeTarget();

	/* Load model from the C array */
	model = tflite::GetModel(g_micro_speech_quantized_model_data);
	if (model->version() != TFLITE_SCHEMA_VERSION) {
		LOG_ERR("Model schema version %d != supported %d",
			model->version(), TFLITE_SCHEMA_VERSION);
		return -1;
	}

	/*
	 * Register only the ops the micro_speech model uses.
	 * The model uses: DepthwiseConv2D, FullyConnected, Softmax, Reshape.
	 */
	static tflite::MicroMutableOpResolver<4> resolver;
	resolver.AddDepthwiseConv2D();
	resolver.AddFullyConnected();
	resolver.AddSoftmax();
	resolver.AddReshape();

	/* Build interpreter */
	static tflite::MicroInterpreter static_interpreter(
		model, resolver, tensor_arena, kTensorArenaSize);
	interpreter = &static_interpreter;

	/* Allocate tensors */
	TfLiteStatus status = interpreter->AllocateTensors();
	if (status != kTfLiteOk) {
		LOG_ERR("AllocateTensors() failed");
		return -2;
	}

	/* Record actual arena usage before any tensor access */
	tflm_arena_used = interpreter->arena_used_bytes();

	input_tensor = interpreter->input(0);
	output_tensor = interpreter->output(0);

	/* Export resolved addresses for debug probe plugin */
	tflm_input_buffer = input_tensor->data.int8;
	tflm_output_buffer = output_tensor->data.int8;
	tflm_input_size = INPUT_SIZE;
	tflm_output_size = CATEGORY_COUNT;
	tflm_output_scale = output_tensor->params.scale;
	tflm_output_zero_point = output_tensor->params.zero_point;

	/* Copy dims to aligned locals to avoid unaligned access faults */
	int in_ndims = input_tensor->dims->size;
	int in_d0 = in_ndims > 0 ? input_tensor->dims->data[0] : 0;
	int in_d1 = in_ndims > 1 ? input_tensor->dims->data[1] : 0;
	int in_d2 = in_ndims > 2 ? input_tensor->dims->data[2] : 0;
	int out_ndims = output_tensor->dims->size;
	int out_d0 = out_ndims > 0 ? output_tensor->dims->data[0] : 0;
	int out_d1 = out_ndims > 1 ? output_tensor->dims->data[1] : 0;

	LOG_INF("Model loaded: %d bytes", g_micro_speech_quantized_model_data_size);
	LOG_INF("Input:  dims=%d shape=[%d,%d,%d] type=%d",
		in_ndims, in_d0, in_d1, in_d2, input_tensor->type);
	LOG_INF("Output: dims=%d shape=[%d,%d] type=%d",
		out_ndims, out_d0, out_d1, output_tensor->type);
	LOG_INF("Arena: %d / %d bytes used", tflm_arena_used, kTensorArenaSize);

	return 0;
}

extern "C" int micro_speech_infer(struct inference_result *result)
{
	if (!interpreter || !input_tensor || !output_tensor) {
		return -1;
	}

	/* Measure inference time */
	uint32_t start = k_cycle_get_32();

	TfLiteStatus status = interpreter->Invoke();

	uint32_t end = k_cycle_get_32();
	uint32_t cycles = end - start;
	result->inference_time_us = k_cyc_to_us_floor32(cycles);

	if (status != kTfLiteOk) {
		LOG_ERR("Invoke() failed");
		return -2;
	}

	/* Dequantize output scores */
	float scale = output_tensor->params.scale;
	int32_t zero_point = output_tensor->params.zero_point;

	int best_class = 0;
	float best_score = -1.0f;

	for (int i = 0; i < CATEGORY_COUNT; i++) {
		int8_t raw = output_tensor->data.int8[i];
		result->scores[i] = (raw - zero_point) * scale;
		if (result->scores[i] > best_score) {
			best_score = result->scores[i];
			best_class = i;
		}
	}

	result->predicted_class = best_class;
	result->confidence = best_score;

	return 0;
}

extern "C" int8_t *micro_speech_get_input_buffer(void)
{
	if (!input_tensor) {
		return nullptr;
	}
	return input_tensor->data.int8;
}

extern "C" int8_t *micro_speech_get_output_buffer(void)
{
	if (!output_tensor) {
		return nullptr;
	}
	return output_tensor->data.int8;
}

extern "C" uint8_t *micro_speech_get_arena(uint32_t *size)
{
	if (size) {
		*size = kTensorArenaSize;
	}
	return tensor_arena;
}

extern "C" void micro_speech_generate_test_input(int pattern)
{
	if (!input_tensor) {
		return;
	}

	int8_t *buf = input_tensor->data.int8;

	switch (pattern) {
	case 0: /* Silence — all zeros (quantized) */
		memset(buf, input_tensor->params.zero_point, INPUT_SIZE);
		break;
	case 1: /* Noise — random-ish uniform values */
		for (int i = 0; i < INPUT_SIZE; i++) {
			/* Simple deterministic pseudo-random */
			buf[i] = (int8_t)((i * 37 + 13) & 0xFF);
		}
		break;
	case 2: /* "Yes"-like — energy concentrated in specific freq bands */
		memset(buf, input_tensor->params.zero_point, INPUT_SIZE);
		for (int t = 10; t < 35; t++) {
			for (int f = 5; f < 25; f++) {
				buf[t * INPUT_FEATURES + f] = 80;
			}
		}
		break;
	case 3: /* "No"-like — different energy pattern */
		memset(buf, input_tensor->params.zero_point, INPUT_SIZE);
		for (int t = 15; t < 40; t++) {
			for (int f = 10; f < 30; f++) {
				buf[t * INPUT_FEATURES + f] = 60;
			}
		}
		break;
	default:
		memset(buf, 0, INPUT_SIZE);
		break;
	}
}

extern "C" const char *micro_speech_get_label(int category)
{
	if (category < 0 || category >= CATEGORY_COUNT) {
		return "invalid";
	}
	return category_labels[category];
}
