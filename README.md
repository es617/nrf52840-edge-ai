# Peripheral UART + Edge AI

This project extends Nordic's [Peripheral UART](README.rst) Zephyr sample with on-device keyword spotting using TFLite Micro. The entire integration — from zero to working inference — was done in a single [Claude Code](https://claude.com/claude-code) session using the [debug probe MCP server](https://github.com/es617/dbgprobe-mcp-server). See the [blog post](https://es617.github.io/2026/03/16/edge-ai-mcp.html) for the full story.

## What it does

A TFLite Micro keyword spotting model ([micro_speech](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech)) runs on an nRF52840 DK alongside the existing BLE UART functionality. Audio is injected via the debug probe, preprocessed on-device (MFCC feature extraction), and classified into four categories: *yes*, *no*, *silence*, *unknown*.

| Metric | Value |
|--------|-------|
| Preprocessing | 67 ms |
| Inference | 31 ms |
| **Total end-to-end** | **98 ms** |
| Flash | 442 KB (42% of 1 MB) |
| RAM | 120 KB (46% of 256 KB) |
| Accuracy | 94.6% (130 samples, Speech Commands v0.02) |

## Quickstart

See [EDGE_AI_SETUP.md](EDGE_AI_SETUP.md) for detailed build instructions, troubleshooting, optimization history, and per-step profiling data.

```bash
# Build (NCS v3.2.2 toolchain must be in PATH)
bash build.sh

# Flash via debug probe
dbgprobe.flash(file="build/peripheral_uart/zephyr/zephyr.hex")

# Run inference on a .wav file
tflite_micro.write_pcm(file="test_data/yes/1.wav")
# → {"predicted_class": "yes", "confidence": 99.6%}
```

## Debug probe plugins

Two plugins turn the debug probe into an edge-AI development environment. Both work with any TFLite Micro model on any Cortex-M target — class labels, model symbols, and CPU frequency are configurable.

- **TFLite Micro inspector** (9 tools) — model inspection, audio injection, inference, batch accuracy testing
- **Cortex-M profiler** (3 tools) — non-invasive cycle counting via DWT hardware counters

Also available in the [dbgprobe-mcp-server](https://github.com/es617/dbgprobe-mcp-server) examples.

## License

This project contains code under multiple licenses:

| Component | License | Details |
|-----------|---------|---------|
| Peripheral UART sample | [Nordic 5-Clause](https://developer.nordicsemi.com/nRF_Connect_SDK/doc/latest/nrf/licenses.html) | Nordic's original code. **Restricted to Nordic hardware.** |
| TFLite Micro | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | Model, runtime, signal library |
| Test data | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | Google Speech Commands v0.02 |
| Edge AI integration (`src/tflm/*`, `scripts/*`) | [MIT](https://opensource.org/licenses/MIT) | TFLite integration, preprocessing, build scripts |
| Debug probe plugins (`.dbgprobe_mcp/plugins/*`) | [MIT](https://opensource.org/licenses/MIT) | Also in [dbgprobe-mcp-server](https://github.com/es617/dbgprobe-mcp-server) |
