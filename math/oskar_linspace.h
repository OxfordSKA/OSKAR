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


#ifndef OSKAR_LINSPACE_H_
#define OSKAR_LINSPACE_H_

/**
 * @file oskar_linspace.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif


/**
* @brief Populates an array with linearly spaced values (Double precision).
 *
 * @details
 * Generates an array of \p n linearly spaced values between \p a and \p b.
 * This function is equivalent to the MATLAB linspace() function.
 * Warning: The \p values array must be preallocated to length \p n.
 *
 * @param[out] values Array of linearly spaced values.
 * @param[in]  a      Start value.
 * @param[in]  b      End value.
 * @param[in]  n      Number of values.
 */
OSKAR_EXPORT
void oskar_linspace_d(double* values, double a, double b, int n);

/**
 * @brief Populates an array with linearly spaced values (Single precision).
 *
 * @details
 * Generates an array of \p n linearly spaced values between \p a and \p b.
 * This function is equivalent to the MATLAB linspace() function.
 * Warning: The \p values array must be preallocated to length \p n.
 *
 * @param[out] values Array of linearly spaced values.
 * @param[in]  a      Start value.
 * @param[in]  b      End value.
 * @param[in]  n      Number of values.
 */
OSKAR_EXPORT
void oskar_linspace_f(float* values, float a, float b, int n);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_LINSPACE_H_ */
