#!/bin/bash
# Build script for peripheral_uart with TFLite Micro
# Source the NCS toolchain environment and run west build.
#
# Usage: source build.sh
#   or:  bash build.sh

TC=/opt/nordic/ncs/toolchains/e5f4758bcf
export PATH="$TC/opt/zephyr-sdk/arm-zephyr-eabi/bin:$TC/opt/bin:$TC/bin:/usr/bin:/bin:$PATH"
export ZEPHYR_TOOLCHAIN_VARIANT=zephyr
export ZEPHYR_SDK_INSTALL_DIR="$TC/opt/zephyr-sdk"
export ZEPHYR_BASE=/opt/nordic/ncs/v3.2.2/zephyr
export NRFUTIL_HOME="$TC/nrfutil/home"
export GIT_EXEC_PATH="$TC/Cellar/git/2.37.3/libexec/git-core"
export GIT_TEMPLATE_DIR="$TC/Cellar/git/2.37.3/share/git-core/templates"

cd "$(dirname "$0")"
$TC/bin/python3 -m west build -b nrf52840dk/nrf52840 "$@"
