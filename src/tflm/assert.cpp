// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Enrico Santagati

/*
 * Override __assert_func for TFLite Micro.
 * TFLite uses C++ assert which calls __assert_func, but Zephyr
 * provides its own assert mechanism.
 */

#include <zephyr/kernel.h>

extern "C" void __assert_func(const char *file, int line,
			       const char *func, const char *failedexpr)
{
	printk("ASSERTION FAIL [%s] @ %s:%d (%s)\n",
	       failedexpr, file, line, func);
	k_panic();
}
