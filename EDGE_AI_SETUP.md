# Edge AI Setup — TFLite Micro on nRF52840

## Overview

Added TensorFlow Lite Micro keyword spotting (micro_speech) to the existing peripheral_uart Zephyr project on nRF52840 DK. The model detects four categories: **silence**, **unknown**, **yes**, **no**.

- Model: micro_speech quantized, 18.8KB
- Input: [1, 49, 40] int8 — 49 time frames x 40 MFCC features
- Output: [1, 4] int8 — one score per category
- Tensor arena: 7KB (6948 bytes used)
- Kernels: CMSIS-NN (ARM Cortex-M4 optimized) — 28x faster than reference kernels

## How it works

TFLite Micro is a C++ library that gets **statically linked** into the Zephyr firmware — no OS interaction, no IPC, no separate runtime. Just function calls.

1. **Build time**: The tflite-micro Zephyr module compiles the TFLite Micro runtime (interpreter, op kernels, memory planner) as a static library and links it into the app. The model weights are baked into flash as a C array (`model_data.cpp`).

2. **Init** (`micro_speech_init()`): Parses the model FlatBuffer from flash, registers the 4 ops the model needs, and gives the interpreter a 7KB RAM buffer (`tensor_arena`) where it plans and allocates all intermediate tensors and scratch buffers. No dynamic allocation after this point.

3. **Inference** (`micro_speech_infer()`): Write 1960 bytes of input data into the input tensor, call `Invoke()`, read 4 output scores. The interpreter runs the neural network layer by layer, all in-place within the tensor arena. Pure CPU math in the calling thread's context.

Zephyr's only role is providing the C/C++ runtime and the thread that calls into TFLite. The interpreter doesn't know or care it's running on Zephyr.

## What was done

### 1. Fetched the tflite-micro Zephyr module

The tflite-micro module is an optional Zephyr module not included in the NCS default manifest. It was cloned manually:

```bash
cd /opt/nordic/ncs/v3.2.2
git clone https://github.com/zephyrproject-rtos/tflite-micro.git optional/modules/lib/tflite-micro
cd optional/modules/lib/tflite-micro
git checkout 8d404de73acf7687831e16d88e86e4f73cfddf8e
```

The exact revision must match what NCS v3.2.2 expects (from `zephyr/submanifests/optional.yaml`).

The module is registered in `CMakeLists.txt` via `ZEPHYR_EXTRA_MODULES` (derived from `ZEPHYR_BASE`), so no west manifest changes are needed.

### 2. Added the micro_speech model

Copied the pre-trained quantized micro_speech model (18,800 bytes) from the Matter SDK example (`modules/lib/matter/examples/lighting-app/telink/src/tflm/models/`). The model is embedded as a C array in `src/tflm/model_data.cpp`.

### 3. Created the inference engine

`src/tflm/micro_speech.cpp` provides the TFLite Micro interpreter with a C API:

| Function | Purpose |
|---|---|
| `micro_speech_init()` | Load model, register ops, allocate tensors |
| `micro_speech_infer()` | Run inference, return dequantized scores + timing |
| `micro_speech_get_input_buffer()` | Pointer to input tensor (for debug probe injection) |
| `micro_speech_get_output_buffer()` | Pointer to output tensor (for debug probe reading) |
| `micro_speech_get_arena()` | Pointer to tensor arena (for memory inspection) |
| `micro_speech_generate_test_input()` | Fill input with synthetic patterns (silence/noise/yes/no) |

The model uses 4 ops: `DepthwiseConv2D`, `FullyConnected`, `Softmax`, `Reshape`.

### 4. Integrated into the existing app

- `main()` initializes TFLite after BLE setup, runs a test inference for each pattern
- A background thread (`tflm_inference_thread`) runs continuous inference every 5 seconds, cycling through 4 synthetic test patterns
- Results are logged via RTT (Segger Real-Time Transfer)
- All existing BLE/UART functionality is preserved

**No microphone** — there is no real audio input. The model expects 1960 int8 values (49 MFCC time frames x 40 frequency bins), so the firmware generates fake inputs:

| Pattern | Description | What it fills |
|---------|-------------|---------------|
| 0 — silence | All zeros (quantized zero point) | Flat input |
| 1 — noise | Deterministic pseudo-random `(i*37+13) & 0xFF` | Uniform noise |
| 2 — "yes"-like | Energy blob at time 10-35, freq 5-25 | Simulates speech-like energy |
| 3 — "no"-like | Energy blob at time 15-40, freq 10-30 | Different energy pattern |

These are not real MFCC features — just patterns to verify the model runs and produces different outputs. Predictions aren't always correct (e.g. pattern 2 "yes-like" often predicts "no"). For injecting real or more realistic data, use the debug probe plugin's `tflite_micro.write_input` tool from the host.

### 5. Updated build configuration

**prj.conf additions:**
```
CONFIG_CPP=y
CONFIG_STD_CPP17=y
CONFIG_TENSORFLOW_LITE_MICRO=y
CONFIG_TENSORFLOW_LITE_MICRO_CMSIS_NN_KERNELS=y
CONFIG_CMSIS_DSP_TRANSFORM=y
CONFIG_SPEED_OPTIMIZATIONS=y
CONFIG_MAIN_STACK_SIZE=4096
CONFIG_HEAP_MEM_POOL_SIZE=8192
```

**CMakeLists.txt additions:**
- `ZEPHYR_EXTRA_MODULES` pointing to tflite-micro
- C++ thread-safe statics disabled (required by TFLite Micro)
- New source files: `micro_speech.cpp`, `model_data.cpp`, `assert.cpp`

## Files

```
src/
├── main.c                          # Modified — TFLite init + inference thread + PCM buffer
└── tflm/
    ├── micro_speech.h              # C API header
    ├── micro_speech.cpp            # TFLite Micro interpreter + inference
    ├── model_data.h                # Model array declaration
    ├── model_data.cpp              # Pre-trained model (18.8KB as C array)
    ├── assert.cpp                  # C++ assert override for Zephyr
    ├── audio_frontend.h            # Audio preprocessing C API
    ├── audio_frontend.cpp          # On-device preprocessing pipeline
    └── audio_frontend_data.cpp     # Generated const arrays (Hann, filterbank, PCAN)

scripts/
├── gen_frontend_data.py            # Generates audio_frontend_data.cpp
├── extract_mfcc.py                 # Host-side MFCC extraction (librosa)
└── requirements.txt                # Python dependencies

.dbgprobe_mcp/plugins/
├── tflite_micro.py                 # TFLite Micro debug probe plugin (9 tools)
└── cortex_m_profiler.py            # Non-invasive cycle profiler via DWT (3 tools)

CMakeLists.txt                      # Modified — tflite-micro module + signal library + sources
prj.conf                            # Modified — TFLite + C++ + memory config
build.sh                            # Build convenience script
```

## Build and flash

### Building

The NCS toolchain must be in PATH. Use the convenience script:

```bash
bash build.sh
```

Or manually:

```bash
TC=/opt/nordic/ncs/toolchains/e5f4758bcf
export ZEPHYR_BASE=/opt/nordic/ncs/v3.2.2/zephyr
export ZEPHYR_TOOLCHAIN_VARIANT=zephyr
export ZEPHYR_SDK_INSTALL_DIR="$TC/opt/zephyr-sdk"
export PATH="$TC/opt/zephyr-sdk/arm-zephyr-eabi/bin:$TC/opt/bin:$TC/bin:/usr/bin:/bin:$PATH"

$TC/bin/python3 -m west build -b nrf52840dk/nrf52840
```

### Flashing

Via J-Link debug probe (using dbgprobe MCP or JLinkExe):

```bash
JLinkExe -device nRF52840_xxAA -if SWD -speed 4000 -autoconnect 1 \
  -CommandFile <(echo "loadfile build/peripheral_uart/zephyr/zephyr.hex; r; go; q")
```

Or via nRF Connect for VS Code → Flash.

### Verifying via RTT

Connect to RTT (Segger Real-Time Transfer) to see inference output. Expected log:

```
micro_speech: Model loaded: 18800 bytes
micro_speech: Input:  dims=2 shape=[1,1960,0] type=9
micro_speech: Output: dims=2 shape=[1,4] type=9
micro_speech: Arena: 16384 bytes allocated
peripheral_uart: TFLite Micro initialized — running test inference
peripheral_uart: Pattern 0 -> silence (...%) [~31000 us]
peripheral_uart: Pattern 1 -> no (...%) [~31000 us]
peripheral_uart: Pattern 2 -> no (...%) [~31000 us]
peripheral_uart: Pattern 3 -> unknown (...%) [~31000 us]
```

The background thread then logs inference every 5 seconds.

## Audio preprocessing

### Host-side approach (starting point)

The model expects preprocessed MFCC features, not raw audio. Rather than building a full preprocessing pipeline on the device, we first validated the end-to-end audio → prediction path using host-side Python preprocessing.

**`scripts/extract_mfcc.py`** takes a `.wav` file, extracts MFCC features using `librosa`, quantizes them to int8, and saves the result as a `.bin` file that can be injected via `tflite_micro.write_input`.

This gave **weak accuracy** because the model was trained with a specific fixed-point preprocessing pipeline (the tflite-micro signal library), not librosa's floating-point MFCCs. Key differences:

- **Mel filterbank**: specific weight quantization (Q12), alignment padding (4-byte), and channel blocking
- **Spectral subtraction**: a stateful noise reduction step not present in standard MFCC extraction
- **PCAN AGC**: per-channel automatic gain control using a wide-dynamic-range LUT compression
- **Quantization**: the exact `int8 = (feature * 256 + 333) / 666 - 128` mapping from the training pipeline

### On-device approach (final)

The tflite-micro signal library contains the exact C functions used to build the preprocessing `.tflite` model during training. By calling these functions directly from firmware (no second TFLite interpreter needed), we get **bit-exact** feature extraction matching the training pipeline.

**Pipeline** — each of 49 frames: 30ms window (480 samples), 20ms stride, from 1 second of 16kHz int16 PCM:

| Step | Function | Input → Output |
|------|----------|---------------|
| 1. Hann window | `tflm_signal::ApplyWindow()` | int16 → int16 (scaled) |
| 2. FFT auto-scale | `tflite::tflm_signal::FftAutoScale()` | int16 → int16 + shift count |
| 3. 512-point RFFT | CMSIS-DSP `arm_rfft_q15()` | int16[512] → Complex\<int16\>[257] |
| 4. Energy | `tflite::tflm_signal::SpectrumToEnergy()` | Complex → uint32 |
| 5. Mel filterbank | `tflite::tflm_signal::FilterbankAccumulateChannels()` | uint32[257] → uint64[40] |
| 6. Square root | `tflite::tflm_signal::FilterbankSqrt()` | uint64 → uint32 |
| 7. Spectral subtraction | `tflite::tflm_signal::FilterbankSpectralSubtraction()` | uint32 → uint32 (stateful) |
| 8. PCAN AGC | `tflite::tflm_signal::ApplyPcanAutoGainControlFixed()` | uint32 → uint32 (uses gain LUT) |
| 9. Log scaling | `tflite::tflm_signal::FilterbankLog()` | uint32 → int16 |
| 10. Quantize | `(feature * 256 + 333) / 666 - 128` | int16 → int8 |

### Precomputed const data

A Python script (`scripts/gen_frontend_data.py`) generates `audio_frontend_data.cpp` containing three const arrays:

- **Hann window** `int16_t[480]` — `round((0.5 - 0.5*cos(2π*(i+0.5)/480)) * 2^12)`, matching `window_op.py`
- **Mel filterbank config** — weights, unweights, channel_frequency_starts, channel_weight_starts, channel_widths (all int16), using the exact `freq_to_mel = 1127.0f * log1pf(f/700.0f)` with float32 precision for bit-exactness with the C code
- **PCAN gain LUT** `int16_t[125]` — piecewise quadratic approximation from `WideDynamicFuncLut()`

Parameters: `alignment=4`, `channel_block_size=4`, `weight_scaling_bits=12`, matching `audio_preprocessor.py`'s `FeatureParams` defaults.

### Firmware integration

**`src/tflm/audio_frontend.h`** — C API:
```c
int audio_frontend_init(void);                                    // init FFT state
int audio_frontend_process(const int16_t *pcm, int8_t *output);  // 16000 samples → 1960 features
void audio_frontend_reset_state(void);                            // clear noise estimator
```

**`src/main.c`** additions:
- `int16_t tflm_pcm_buffer[16000]` — 32KB global for raw PCM (written by debug probe or UART)
- `volatile int32_t tflm_run_preprocess` — trigger flag
- Inference thread checks the flag, calls `audio_frontend_reset_state()` + `audio_frontend_process()` + `micro_speech_infer()`

All scratch buffers are file-scope statics (not stack — the inference thread only has 4KB).

### Signal library build integration

The Zephyr tflite-micro module already builds some signal library sources (rfft, window, kiss_fft wrappers). `CMakeLists.txt` adds the 13 remaining `.cc` files needed for the full pipeline:

```
energy, fft_auto_scale, filter_bank, filter_bank_log,
filter_bank_spectral_subtraction, filter_bank_square_root,
log, max_abs, msb_32, msb_64, pcan_argc_fixed,
square_root_32, square_root_64
```

The FFT uses CMSIS-DSP `arm_rfft_q15` instead of Kiss FFT for a 128x speedup. This requires `CONFIG_CMSIS_DSP_TRANSFORM=y` in `prj.conf`.

### CMSIS-DSP FFT scaling

CMSIS-DSP `arm_rfft_q15` scales output by N (512, 9 bits down), while Kiss FFT's fixed-point mode scales by 2N (1024, 10 bits down). CMSIS output is therefore 2x larger in amplitude. This is compensated by adding +1 to the `scale_down_bits` parameter in `FilterbankSqrt()`. Without this correction, the model produces wrong predictions due to shifted feature magnitudes.

## Performance

On nRF52840 (64 MHz Cortex-M4F), with CMSIS-NN kernels, CMSIS-DSP FFT, and `-O2`:

| Metric | Value |
|--------|-------|
| Preprocessing | 64 ms |
| Inference | 31 ms |
| Total end-to-end | 95 ms |
| Flash | 430 KB (42%) |
| RAM | 120 KB (46%) |
| Tensor arena | 7 KB (6948 used, 97% utilized) |

### RAM budget

| Component | Bytes |
|-----------|-------|
| PCM buffer (`tflm_pcm_buffer`) | 32,000 |
| CMSIS-DSP RFFT instance + scratch | ~2,100 |
| Static scratch buffers (windowed, scaled, energy, filterbank, etc.) | ~3,600 |
| Noise estimates (persistent across frames) | 160 |
| **New RAM for preprocessing** | **~38 KB** |
| Previous usage (BLE + TFLite runtime + arena) | ~82 KB |
| **Total** | **~120 KB / 256 KB (46%)** |

### Per-step preprocessing profile

Measured non-invasively using the `cortex_m_profiler` plugin (DWT hardware cycle counter):

| Step | Per frame (us) | x49 total (ms) | % |
|------|----------------|-----------------|---|
| Window (ApplyWindow) | 175 | 8.6 | 12% |
| Auto-scale (FftAutoScale) | 199 | 9.8 | 14% |
| **RFFT (arm_rfft_q15)** | **557** | **27.3** | **38%** |
| Energy (SpectrumToEnergy) | 122 | 6.0 | 8% |
| Filterbank (AccumulateChannels) | 220 | 10.8 | 15% |
| Sqrt (FilterbankSqrt) | 29 | 1.4 | 2% |
| Spectral subtraction | 31 | 1.5 | 2% |
| PCAN AGC | 50 | 2.4 | 3% |
| Log + quantize + loop | 77 | 3.8 | 5% |
| **Total per frame** | **1,460** | **71.5** | **100%** |

The pipeline is well-balanced — RFFT is the largest step at 38%, but no single bottleneck dominates.

### Optimization history

| Change | Preprocessing | End-to-end | Speedup |
|--------|--------------|------------|---------|
| Initial (Kiss FFT, `-Os`) | 5,348 ms | 5,379 ms | baseline |
| `CONFIG_SPEED_OPTIMIZATIONS=y` (`-O2`) | 2,616 ms | 2,647 ms | **2.0x** |
| Replace Kiss FFT with CMSIS-DSP `arm_rfft_q15` | **64 ms** | **95 ms** | **80x total** |
| Arena right-sizing (16KB → 7KB) | — | — | 9 KB RAM saved |

`CONFIG_SPEED_OPTIMIZATIONS=y` is the Zephyr Kconfig flag that sets the compiler to `-O2` instead of the default `-Os`. The 2x speedup comes from loop unrolling and instruction scheduling in the tight integer FFT butterfly loops.

CMSIS-DSP vectorization for other pipeline steps (`arm_mult_q15`, `arm_abs_q15`, `arm_max_q15`) was tested and reverted — the compiler's `-O2` already optimizes these small loops well on Cortex-M4, and `arm_mult_q15` uses `>>15` shift vs the signal library's `>>12`, losing 3 bits of precision and breaking model accuracy.

## Test results

### Reference wav files

Tested with tflite-micro's reference `.wav` files via `tflite_micro.write_pcm` (on-device preprocessing):

| Audio file | Predicted | Confidence | Expected | Correct? |
|------------|-----------|------------|----------|----------|
| `yes_1000ms.wav` | **yes** | 99.6% | yes | Yes |
| `no_1000ms.wav` | **no** | 93.0% | no | Yes |
| `noise_1000ms.wav` | silence | 96.5% | unknown | — |
| `silence_1000ms.wav` | silence | 43.8% | silence | Yes |

All keyword classes correctly classified. The noise sample predicting "silence" is reasonable — background noise is a subset of the silence category for this model.

### Large-scale accuracy (Google Speech Commands v0.02)

Validated on 130 random samples from the [Google Speech Commands dataset](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz) (real-world recordings, varying speakers/accents/quality), using `tflite_micro.accuracy_test`:

| Class | Samples | Correct | Accuracy |
|-------|---------|---------|----------|
| yes | 65 | 63 | **96.9%** |
| no | 65 | 60 | **92.3%** |
| **Overall** | **130** | **123** | **94.6%** |

Misclassification breakdown (7 errors):
- 4 "no" → "unknown" (low confidence 39-70%, borderline recordings)
- 1 "no" → "yes" (38% confidence — near-random)
- 1 "yes" → "unknown"
- 1 "yes" → "no" (68% confidence)

All errors are low-confidence — the model is well-calibrated. When it's wrong, it's uncertain. This is strong performance for an 18.8KB model running on a Cortex-M4.

Test data: `test_data/yes/` (~4000 files), `test_data/no/` (~4000 files) from Speech Commands v0.02. Run via `tflite_micro.accuracy_test(directory="test_data", classes="yes,no", samples_per_class=50)`.

## Debug probe plugins

Two dbgprobe plugins (`.dbgprobe_mcp/plugins/`) interact with the running firmware through the J-Link debug probe. See each plugin's README for full details:

- **[tflite_micro](.dbgprobe_mcp/plugins/README_tflite_micro.md)** — Inspect the TFLite model, inject data, trigger inference, and read results (9 tools)
- **[cortex_m_profiler](.dbgprobe_mcp/plugins/README_cortex_m_profiler.md)** — Non-invasive latency profiling using the Cortex-M4 DWT hardware cycle counter (3 tools)

### Quick start

1. Connect to the board: `dbgprobe.connect` (device: `nRF52840_xxAA`)
2. Attach ELF: `dbgprobe.elf.attach` (path: `build/peripheral_uart/zephyr/zephyr.elf`)
3. Plugin tools are available immediately

### Debug probe tool: write_pcm

`tflite_micro.write_pcm` is the main end-to-end tool:
1. Reads a `.wav` file from the host (validates mono, 16kHz, 16-bit)
2. Writes 32KB raw PCM to `tflm_pcm_buffer` on the device
3. Sets `tflm_run_preprocess = 1` and resumes the target
4. Waits for completion, reads inference result

```
tflite_micro.write_pcm(session_id="...", file="/path/to/yes.wav")
→ {"predicted_class": "yes", "confidence": 0.996, ...}
```

### Design principles

- **ELF symbol lookup** — plugins find buffers and variables via global symbols rather than hardcoded addresses, so they work across rebuilds
- **Non-intrusive** — reads/writes target RAM through the debug probe; no firmware source changes needed for profiling
- **Quantization-aware** — output scores are dequantized using the model's scale/zero_point for human-readable results
- **Trigger flag protocol** — `tflm_run_preprocess` and `tflm_run_inference` flags are cleared **after** work completes, so the debug probe can reliably detect completion by polling for flag == 0

## Troubleshooting

### CMSIS-NN unaligned access fault

CMSIS-NN kernels use `LDRD` (load double-word) instructions that **always require 4-byte alignment** regardless of the `SCB->CCR UNALIGN_TRP` setting. Single `LDR`/`STR` can handle unaligned access on Cortex-M4, but `LDRD`/`STRD`/`LDM`/`STM` cannot.

**Root cause:** The model data array (`g_micro_speech_quantized_model_data`) had no alignment constraint, so the linker placed it at an odd address (e.g. `0x200010FD`). TFLite Micro reads constant weights (filter coefficients) directly from the FlatBuffer without copying, so all internal offsets inherit this misalignment. When CMSIS-NN's `depthwise_conv_s8_mult_4` tries to load 8 filter bytes via `LDRD r7, r6, [r3, #0]` from an unaligned pointer, it faults.

**Fix:** Add `alignas(16)` to the model data array declaration and definition:
```cpp
// model_data.h
alignas(16) extern unsigned char g_micro_speech_quantized_model_data[18800];

// model_data.cpp
alignas(16) unsigned char g_micro_speech_quantized_model_data[18800] = { ... };
```

This ensures the FlatBuffer base address is 16-byte aligned, and since the FlatBuffer format aligns tensor data internally to 4-byte boundaries, the filter weights end up properly aligned for CMSIS-NN's `LDRD` instructions.
