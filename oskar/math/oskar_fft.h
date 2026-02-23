/*
 * Copyright (c) 2019-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_FFT_H_
#define OSKAR_FFT_H_

/**
 * @file oskar_fft.h
 */

#include "oskar_global.h"
#include "mem/oskar_mem.h"

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
oskar_FFT* oskar_fft_create(
        int precision,
        int location,
        int num_dim,
        int dim_size,
        int batch_size_1d,
        int* status
);

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

#endif /* include guard */
