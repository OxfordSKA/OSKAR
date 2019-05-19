/*
 * Copyright (c) 2019, The University of Oxford
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

#ifndef OSKAR_FFT_H_
#define OSKAR_FFT_H_

/**
 * @file oskar_fft.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_FFT;
#ifndef OSKAR_FFT_TYPEDEF_
#define OSKAR_FFT_TYPEDEF_
typedef struct oskar_FFT oskar_FFT;
#endif /* OSKAR_FFT_TYPEDEF_ */

/**
 * @brief Create FFT plan.
 *
 * @details
 * Creates a plan for executing FFTs.
 *
 * @param[in] precision     Enumerated data type precision.
 * @param[in] location      Enumerated compute platform.
 * @param[in] num_dim       Number of dimensions.
 * @param[in] dim_size      The size of each dimension.
 * @param[in] batch_size_1d Batch size for 1D transforms.
 * @param[in,out] status    Status return code.
 *
 * @return A pointer to the created plan.
 */
OSKAR_EXPORT
oskar_FFT* oskar_fft_create(int precision, int location, int num_dim,
        int dim_size, int batch_size_1d, int* status);

/**
 * @brief Executes the FFT plan.
 *
 * @details
 * Executes the FFT plan with the supplied data.
 * The transform is done effectively "in-place"
 * (although the details of precisely how the transform is done are
 * implementation-specific and not known in general).
 *
 * @param[in] h             Handle to FFT plan.
 * @param[in] data          Pointer to data to transform.
 * 					        Must be consistent with that specified in
 * 					        oskar_fft_create().
 * @param[in,out] status    Status return code.
 */
OSKAR_EXPORT
void oskar_fft_exec(oskar_FFT* h, oskar_Mem* data, int* status);

/**
 * @brief Frees resources used by the plan.
 *
 * @details
 * Frees resources used by the plan.
 *
 * @param[in] h     Handle to FFT plan.
 */
OSKAR_EXPORT
void oskar_fft_free(oskar_FFT* h);

OSKAR_EXPORT
void oskar_fft_set_ensure_consistent_norm(oskar_FFT* h, int value);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_FFT_H_ */
