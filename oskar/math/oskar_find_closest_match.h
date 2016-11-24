/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#ifndef OSKAR_FIND_CLOSEST_MATCH_H_
#define OSKAR_FIND_CLOSEST_MATCH_H_

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Finds the index of the closest match in an array of \p values to a
 * specified \p value (single precision).
 *
 * @details
 *
 * @param[in]  value       The value to match.
 * @param[in]  values      An array of values to check.
 * @param[in,out] status   Status return code.
 *
 * @return The index of the closest match
 */
OSKAR_EXPORT
int oskar_find_closest_match_f(float value, int num_values,
        const float* values);

/**
 * @brief
 * Finds the index of the closest match in an array of \p values to a
 * specified \p value (double precision).
 *
 * @details
 *
 * @param[in]  value       The value to match.
 * @param[in]  values      An array of values to check.
 * @param[in,out] status   Status return code.
 *
 * @return The index of the closest match
 */
OSKAR_EXPORT
int oskar_find_closest_match_d(double value, int num_values,
        const double* values);

/**
 * @brief
 * Finds the index of the closest match in an array of \p values to a
 * specified \p value.
 *
 * @details
 *
 * @param[in]  value       The value to match.
 * @param[in]  values      An array of values to check.
 * @param[in,out] status   Status return code.
 *
 * @return The index of the closest match
 */
OSKAR_EXPORT
int oskar_find_closest_match(double value, const oskar_Mem* values,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_FIND_CLOSEST_MATCH_H_ */
