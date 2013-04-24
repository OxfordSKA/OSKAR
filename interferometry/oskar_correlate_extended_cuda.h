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

#ifndef OSKAR_CORRELATE_EXTENDED_CUDA_H_
#define OSKAR_CORRELATE_EXTENDED_CUDA_H_

/**
 * @file oskar_correlate_extended_cuda.h
 */

#include "oskar_global.h"
#include "utility/oskar_vector_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * CUDA correlate function for extended Gaussian sources (single precision).
 *
 * @details
 * Forms visibilities on all baselines by correlating Jones matrices for pairs
 * of stations and summing along the source dimension.
 *
 * Gaussian parameters a, b, and c are assumed to be evaluated when the
 * sky model is loaded.
 *
 * Note that all pointers refer to device memory, and must not be dereferenced
 * in host code.
 *
 * @param[in] num_sources   Number of sources.
 * @param[in] num_stations  Number of stations.
 * @param[in] d_jones       Matrix of Jones matrices to correlate.
 * @param[in] d_source_I    Source Stokes I values, in Jy.
 * @param[in] d_source_Q    Source Stokes Q values, in Jy.
 * @param[in] d_source_U    Source Stokes U values, in Jy.
 * @param[in] d_source_V    Source Stokes V values, in Jy.
 * @param[in] d_source_l    Source l-direction cosines from phase centre.
 * @param[in] d_source_m    Source m-direction cosines from phase centre.
 * @param[in] d_source_a    Source Gaussian parameter a.
 * @param[in] d_source_b    Source Gaussian parameter b.
 * @param[in] d_source_c    Source Gaussian parameter c.
 * @param[in] d_station_u   Station u-coordinates multiplied by the wavenumber.
 * @param[in] d_station_v   Station v-coordinates multiplied by the wavenumber.
 * @param[in] freq_hz       Frequency, in Hz.
 * @param[in] bandwidth_hz  Channel bandwidth, in Hz.
 * @param[in,out] d_vis     Modified output complex visibilities.
 */
OSKAR_EXPORT
void oskar_correlate_extended_cuda_f(int num_sources,
        int num_stations, const float4c* d_jones,
        const float* d_source_I, const float* d_source_Q,
        const float* d_source_U, const float* d_source_V,
        const float* d_source_l, const float* d_source_m,
        const float* d_source_a, const float* d_source_b,
        const float* d_source_c, const float* d_station_u,
        const float* d_station_v, float freq_hz, float bandwidth_hz,
        float4c* d_vis);

/**
 * @brief
 * CUDA correlate function for extended Gaussian sources (double precision).
 *
 * @details
 * Forms visibilities on all baselines by correlating Jones matrices for pairs
 * of stations and summing along the source dimension.
 *
 * Gaussian parameters a, b, and c are assumed to be evaluated when the
 * sky model is loaded.
 *
 * Note that all pointers refer to device memory, and must not be dereferenced
 * in host code.
 *
 * @param[in] num_sources   Number of sources.
 * @param[in] num_stations  Number of stations.
 * @param[in] d_jones       Matrix of Jones matrices to correlate.
 * @param[in] d_source_I    Source Stokes I values, in Jy.
 * @param[in] d_source_Q    Source Stokes Q values, in Jy.
 * @param[in] d_source_U    Source Stokes U values, in Jy.
 * @param[in] d_source_V    Source Stokes V values, in Jy.
 * @param[in] d_source_l    Source l-direction cosines from phase centre.
 * @param[in] d_source_m    Source m-direction cosines from phase centre.
 * @param[in] d_source_a    Source Gaussian parameter a.
 * @param[in] d_source_b    Source Gaussian parameter b.
 * @param[in] d_source_c    Source Gaussian parameter c.
 * @param[in] d_station_u   Station u-coordinates multiplied by the wavenumber.
 * @param[in] d_station_v   Station v-coordinates multiplied by the wavenumber.
 * @param[in] freq_hz       Frequency, in Hz.
 * @param[in] bandwidth_hz  Channel bandwidth, in Hz.
 * @param[in,out] d_vis     Modified output complex visibilities.
 */
OSKAR_EXPORT
void oskar_correlate_extended_cuda_d(int num_sources,
        int num_stations, const double4c* d_jones,
        const double* d_source_I, const double* d_source_Q,
        const double* d_source_U, const double* d_source_V,
        const double* d_source_l, const double* d_source_m,
        const double* d_source_a, const double* d_source_b,
        const double* d_source_c, const double* d_station_u,
        const double* d_station_v, double freq_hz, double bandwidth_hz,
        double4c* d_vis);

#ifdef __CUDACC__

/**
 * @brief
 * CUDA correlate kernel for extended Gaussian sources (single precision).
 *
 * @details
 * Forms visibilities on all baselines by correlating Jones matrices for pairs
 * of stations and summing along the source dimension.
 *
 * Gaussian parameters a, b and c are assumed to be evaluated when the
 * sky model is loaded.
 *
 * @param[in] num_sources   Number of sources.
 * @param[in] num_stations  Number of stations.
 * @param[in] jones         Matrix of Jones matrices to correlate.
 * @param[in] source_I      Source Stokes I values, in Jy.
 * @param[in] source_Q      Source Stokes Q values, in Jy.
 * @param[in] source_U      Source Stokes U values, in Jy.
 * @param[in] source_V      Source Stokes V values, in Jy.
 * @param[in] source_l      Source l-direction cosines from phase centre.
 * @param[in] source_m      Source m-direction cosines from phase centre.
 * @param[in] source_a      Source Gaussian parameter a.
 * @param[in] source_b      Source Gaussian parameter b.
 * @param[in] source_c      Source Gaussian parameter c.
 * @param[in] station_u     Station u-coordinates multiplied by the wavenumber.
 * @param[in] station_v     Station v-coordinates multiplied by the wavenumber.
 * @param[in] freq_hz       Frequency, in Hz.
 * @param[in] bandwidth_hz  Channel bandwidth, in Hz.
 * @param[in,out] vis       Modified output complex visibilities.
 */
__global__
void oskar_correlate_extended_cudak_f(const int num_sources,
        const int num_stations, const float4c* jones, const float* source_I,
        const float* source_Q, const float* source_U, const float* source_V,
        const float* source_l, const float* source_m,
        const float* source_a, const float* source_b, const float* source_c,
        const float* station_u, const float* station_v, const float freq_hz,
        const float bandwidth_hz, float4c* vis);

/**
 * @brief
 * CUDA correlate kernel for extended Gaussian sources (double precision).
 *
 * @details
 * Forms visibilities on all baselines by correlating Jones matrices for pairs
 * of stations and summing along the source dimension.
 *
 * Gaussian parameters a, b and c are assumed to be evaluated when the
 * sky model is loaded.
 *
 * @param[in] num_sources   Number of sources.
 * @param[in] num_stations  Number of stations.
 * @param[in] jones         Matrix of Jones matrices to correlate.
 * @param[in] source_I      Source Stokes I values, in Jy.
 * @param[in] source_Q      Source Stokes Q values, in Jy.
 * @param[in] source_U      Source Stokes U values, in Jy.
 * @param[in] source_V      Source Stokes V values, in Jy.
 * @param[in] source_l      Source l-direction cosines from phase centre.
 * @param[in] source_m      Source m-direction cosines from phase centre.
 * @param[in] source_a      Source Gaussian parameter a.
 * @param[in] source_b      Source Gaussian parameter b.
 * @param[in] source_c      Source Gaussian parameter c.
 * @param[in] station_u     Station u-coordinates multiplied by the wavenumber.
 * @param[in] station_v     Station v-coordinates multiplied by the wavenumber.
 * @param[in] freq_hz       Frequency, in Hz.
 * @param[in] bandwidth_hz  Channel bandwidth, in Hz.
 * @param[in,out] vis       Modified output complex visibilities.
 */
__global__
void oskar_correlate_extended_cudak_d(const int num_sources,
        const int num_stations, const double4c* jones, const double* source_I,
        const double* source_Q, const double* source_U, const double* source_V,
        const double* source_l, const double* source_m,
        const double* source_a, const double* source_b, const double* source_c,
        const double* station_u, const double* station_v, const double freq_hz,
        const double bandwidth_hz, double4c* vis);

#endif /* __CUDACC__ */

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CORRELATE_EXTENDED_CUDA_H_ */
