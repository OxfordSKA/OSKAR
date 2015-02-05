/*
 * Copyright (c) 2015, The University of Oxford
 * All rights reserved.
 *
 * This file is part of the OSKAR package.
 * Contact: oskar at oerc.ox.ac.uk
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef UTILITY_OSKAR_GET_MEMORY_USAGE_H_
#define UTILITY_OSKAR_GET_MEMORY_USAGE_H_

/**
 * @file oskar_get_memory_usage.h
 */

#include <oskar_global.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Returns the total physical system memory, in bytes.
 */
OSKAR_EXPORT
size_t oskar_get_total_physical_memory(void);

/**
 * @brief Returns the free physical system memory, in bytes.
 */
OSKAR_EXPORT
size_t oskar_get_free_physical_memory(void);

/**
 * @brief Returns the total swap memory, in bytes.
 */
OSKAR_EXPORT
size_t oskar_get_total_swap_memory(void);

/**
 * @brief Returns the free swap system, in bytes.
 */
OSKAR_EXPORT
size_t oskar_get_free_swap_memory(void);

/**
 * @brief Prints a summary of the current memory usage.
 */
OSKAR_EXPORT
void oskar_print_memory_info(void);


#ifdef __cplusplus
}
#endif

#endif /* UTILITY_OSKAR_GET_MEMORY_USAGE_H_ */
