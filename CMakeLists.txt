#
# Copyright (c) 2018 Nordic Semiconductor
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
cmake_minimum_required(VERSION 3.20.0)

# Add tflite-micro as an extra Zephyr module (not in NCS default manifest)
# Derive NCS root from ZEPHYR_BASE env var (which is <ncs_root>/zephyr)
set(TFLITE_MICRO_DIR "$ENV{ZEPHYR_BASE}/../optional/modules/lib/tflite-micro")
get_filename_component(TFLITE_MICRO_DIR "${TFLITE_MICRO_DIR}" ABSOLUTE)
if(EXISTS "${TFLITE_MICRO_DIR}/zephyr/module.yml")
  list(APPEND ZEPHYR_EXTRA_MODULES "${TFLITE_MICRO_DIR}")
else()
  message(WARNING "tflite-micro module not found at ${TFLITE_MICRO_DIR}")
endif()

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(peripheral_uart)

# C++ settings for TFLite Micro (disable thread-safe statics)
set(NO_THREADSAFE_STATICS $<TARGET_PROPERTY:compiler-cpp,no_threadsafe_statics>)
zephyr_compile_options($<$<COMPILE_LANGUAGE:CXX>:${NO_THREADSAFE_STATICS}>)

# NORDIC SDK APP START
target_sources(app PRIVATE
  src/main.c
)
# NORDIC SDK APP END

# TFLite Micro inference engine
target_sources(app PRIVATE
  src/tflm/micro_speech.cpp
  src/tflm/model_data.cpp
  src/tflm/assert.cpp
)

# Audio preprocessing frontend (signal library pipeline)
target_sources(app PRIVATE
  src/tflm/audio_frontend.cpp
  src/tflm/audio_frontend_data.cpp
)

# Signal library sources not already in the Zephyr tflite-micro module
# (rfft_int16, window, kiss_fft wrappers are already built by the module)
target_sources(app PRIVATE
  ${TFLITE_MICRO_DIR}/signal/src/energy.cc
  ${TFLITE_MICRO_DIR}/signal/src/fft_auto_scale.cc
  ${TFLITE_MICRO_DIR}/signal/src/filter_bank.cc
  ${TFLITE_MICRO_DIR}/signal/src/filter_bank_log.cc
  ${TFLITE_MICRO_DIR}/signal/src/filter_bank_spectral_subtraction.cc
  ${TFLITE_MICRO_DIR}/signal/src/filter_bank_square_root.cc
  ${TFLITE_MICRO_DIR}/signal/src/log.cc
  ${TFLITE_MICRO_DIR}/signal/src/max_abs.cc
  ${TFLITE_MICRO_DIR}/signal/src/msb_32.cc
  ${TFLITE_MICRO_DIR}/signal/src/msb_64.cc
  ${TFLITE_MICRO_DIR}/signal/src/pcan_argc_fixed.cc
  ${TFLITE_MICRO_DIR}/signal/src/square_root_32.cc
  ${TFLITE_MICRO_DIR}/signal/src/square_root_64.cc
)

# Include paths for signal library internal headers
zephyr_include_directories(
  ${TFLITE_MICRO_DIR}/.
  ${TFLITE_MICRO_DIR}/signal/src
)
