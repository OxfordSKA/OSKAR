/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#ifndef OSKAR_EVALUATE_JONES_K_CUDA_H_
#define OSKAR_EVALUATE_JONES_K_CUDA_H_

/**
 * @file oskar_evaluate_jones_K_cuda.h
 */

#include <oskar_global.h>
#include <utility/oskar_vector_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates the interferometer phase (K) Jones term using CUDA
 * (single precision).
 *
 * @details
 * This function constructs a set of Jones matrices that correspond to the
 * interferometer phase offset for each source and station, relative to the
 * array centre.
 *
 * The output set of Jones matrices (K) are scalar complex values.
 *
 * @param[out] d_jones      Output set of Jones matrices.
 * @param[in]  num_sources  Number of sources.
 * @param[in]  d_l          Source l-direction cosines.
 * @param[in]  d_m          Source m-direction cosines.
 * @param[in]  d_n          Source n-direction cosines.
 * @param[in]  num_stations Number of stations.
 * @param[in]  d_u          Station u coordinates, in metres.
 * @param[in]  d_v          Station v coordinates, in metres.
 * @param[in]  d_w          Station w coordinates, in metres.
 * @param[in]  wavenumber   Wavenumber (2 pi / wavelength).
 */
OSKAR_EXPORT
void oskar_evaluate_jones_K_cuda_f(float2* d_jones, int num_sources,
        const float* d_l, const float* d_m, const float* d_n,
        int num_stations, const float* d_u, const float* d_v,
        const float* d_w, float wavenumber, const float* d_source_filter,
        float source_filter_min, float source_filter_max);

/**
 * @brief
 * Evaluates the interferometer phase (K) Jones term using CUDA
 * (double precision).
 *
 * @details
 * This function constructs a set of Jones matrices that correspond to the
 * interferometer phase offset for each source and station, relative to the
 * array centre.
 *
 * The output set of Jones matrices (K) are scalar complex values.
 *
 * @param[out] d_jones      Output set of Jones matrices.
 * @param[in]  num_sources  Number of sources.
 * @param[in]  d_l          Source l-direction cosines.
 * @param[in]  d_m          Source m-direction cosines.
 * @param[in]  d_n          Source n-direction cosines.
 * @param[in]  num_stations Number of stations.
 * @param[in]  d_u          Station u coordinates, in metres.
 * @param[in]  d_v          Station v coordinates, in metres.
 * @param[in]  d_w          Station w coordinates, in metres.
 * @param[in]  wavenumber   Wavenumber (2 pi / wavelength).
 */
OSKAR_EXPORT
void oskar_evaluate_jones_K_cuda_d(double2* d_jones, int num_sources,
        const double* d_l, const double* d_m, const double* d_n,
        int num_stations, const double* d_u, const double* d_v,
        const double* d_w, double wavenumber, const double* d_source_filter,
        double source_filter_min, double source_filter_max);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_JONES_K_CUDA_H_ */
