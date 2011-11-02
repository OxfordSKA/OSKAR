/*
 * Copyright (c) 2011, The University of Oxford
 * All rights reserved.
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

#ifndef OSKAR_STRING_TO_ARRAY_H_
#define OSKAR_STRING_TO_ARRAY_H_

/**
 * @file oskar_string_to_array.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Splits a string into numeric fields (single precision).
 *
 * @details
 * This function splits a string into a sequence of numbers. Splitting is
 * performed either using whitespace or a comma.
 *
 * <b>Note that the input string is corrupted on exit.</b>
 *
 * @param[in,out] str The input string to split.
 * @param[in] n The maximum number of values to return (size of array \p data).
 * @param[out] data The array of values returned.
 * @param[in] save_ptr A pointer to use between calls.
 *
 * @return The number of values matched (or number of array elements filled).
 */
OSKAR_EXPORT
int oskar_string_to_array_f(char* str, int n, float* data);

/**
 * @brief Splits a string into numeric fields (double precision).
 *
 * @details
 * This function splits a string into a sequence of numbers. Splitting is
 * performed either using whitespace or a comma.
 *
 * <b>Note that the input string is corrupted on exit.</b>
 *
 * @param[in,out] str The input string to split.
 * @param[in] n The maximum number of values to return (size of array \p data).
 * @param[out] data The array of values returned.
 * @param[in] save_ptr A pointer to use between calls.
 *
 * @return The number of values matched (or number of array elements filled).
 */
OSKAR_EXPORT
int oskar_string_to_array_d(char* str, int n, double* data);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_STRING_TO_ARRAY_H_ */
