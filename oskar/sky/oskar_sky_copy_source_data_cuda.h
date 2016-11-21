/*
 * Copyright (c) 2014, The University of Oxford
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

#ifndef OSKAR_SKY_COPY_SOURCE_DATA_CUDA_H_
#define OSKAR_SKY_COPY_SOURCE_DATA_CUDA_H_

/**
 * @file oskar_sky_copy_source_data_cuda.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Copies source data to new arrays based on mask values (single precision).
 *
 * @details
 * Copies source data to new arrays wherever the input mask value is positive.
 *
 * Each of the output arrays must be sized large enough to hold all the input
 * data if necessary.
 *
 * Note that all pointers refer to device memory.
 *
 * @param[in] num       Number of input sources.
 * @param[out] num_out  Number of sources copied.
 * @param[in] mask      Input mask values.
 * @param[in] ra_in     Input right ascension coordinates.
 * @param[out] ra_out   Output right ascension coordinates.
 * @param[in] dec_in    Input declination coordinates.
 * @param[out] dec_out  Output declination coordinates.
 * @param[in] I_in      Input Stokes I values.
 * @param[out] I_out    Output Stokes I values.
 * @param[in] Q_in      Input Stokes Q values.
 * @param[out] Q_out    Output Stokes Q values.
 * @param[in] U_in      Input Stokes U values.
 * @param[out] U_out    Output Stokes U values.
 * @param[in] V_in      Input Stokes V values.
 * @param[out] V_out    Output Stokes V values.
 * @param[in] ref_in    Input reference frequency values.
 * @param[out] ref_out  Output reference frequency values.
 * @param[in] sp_in     Input spectral index values.
 * @param[out] sp_out   Output spectral index values.
 * @param[in] rm_in     Input rotation measure values.
 * @param[out] rm_out   Output rotation measure values.
 * @param[in] l_in      Input l-direction-cosine values.
 * @param[out] l_out    Output l-direction-cosine values.
 * @param[in] m_in      Input m-direction-cosine values.
 * @param[out] m_out    Output m-direction-cosine values.
 * @param[in] n_in      Input n-direction-cosine values.
 * @param[out] n_out    Output n-direction-cosine values.
 * @param[in] a_in      Input Gaussian parameter a values.
 * @param[out] a_out    Output Gaussian parameter a values.
 * @param[in] b_in      Input Gaussian parameter b values.
 * @param[out] b_out    Output Gaussian parameter b values.
 * @param[in] c_in      Input Gaussian parameter c values.
 * @param[out] c_out    Output Gaussian parameter c values.
 * @param[in] maj_in    Input Gaussian major axis values.
 * @param[out] maj_out  Output Gaussian major axis values.
 * @param[in] min_in    Input Gaussian minor axis values.
 * @param[out] min_out  Output Gaussian minor axis values.
 * @param[in] pa_in     Input Gaussian position angle values.
 * @param[out] pa_out   Output Gaussian position angle values.
 */
OSKAR_EXPORT
void oskar_sky_copy_source_data_cuda_f(
        int num, int* num_out, const int* mask,
        const float* ra_in,   float* ra_out,
        const float* dec_in,  float* dec_out,
        const float* I_in,    float* I_out,
        const float* Q_in,    float* Q_out,
        const float* U_in,    float* U_out,
        const float* V_in,    float* V_out,
        const float* ref_in,  float* ref_out,
        const float* sp_in,   float* sp_out,
        const float* rm_in,   float* rm_out,
        const float* l_in,    float* l_out,
        const float* m_in,    float* m_out,
        const float* n_in,    float* n_out,
        const float* a_in,    float* a_out,
        const float* b_in,    float* b_out,
        const float* c_in,    float* c_out,
        const float* maj_in,  float* maj_out,
        const float* min_in,  float* min_out,
        const float* pa_in,   float* pa_out
        );

/**
 * @brief
 * Copies source data to new arrays based on mask values (double precision).
 *
 * @details
 * Copies source data to new arrays wherever the input mask value is positive.
 *
 * Each of the output arrays must be sized large enough to hold all the input
 * data if necessary.
 *
 * Note that all pointers refer to device memory.
 *
 * @param[in] num       Number of input sources.
 * @param[out] num_out  Number of sources copied.
 * @param[in] mask      Input mask values.
 * @param[in] ra_in     Input right ascension coordinates.
 * @param[out] ra_out   Output right ascension coordinates.
 * @param[in] dec_in    Input declination coordinates.
 * @param[out] dec_out  Output declination coordinates.
 * @param[in] I_in      Input Stokes I values.
 * @param[out] I_out    Output Stokes I values.
 * @param[in] Q_in      Input Stokes Q values.
 * @param[out] Q_out    Output Stokes Q values.
 * @param[in] U_in      Input Stokes U values.
 * @param[out] U_out    Output Stokes U values.
 * @param[in] V_in      Input Stokes V values.
 * @param[out] V_out    Output Stokes V values.
 * @param[in] ref_in    Input reference frequency values.
 * @param[out] ref_out  Output reference frequency values.
 * @param[in] sp_in     Input spectral index values.
 * @param[out] sp_out   Output spectral index values.
 * @param[in] rm_in     Input rotation measure values.
 * @param[out] rm_out   Output rotation measure values.
 * @param[in] l_in      Input l-direction-cosine values.
 * @param[out] l_out    Output l-direction-cosine values.
 * @param[in] m_in      Input m-direction-cosine values.
 * @param[out] m_out    Output m-direction-cosine values.
 * @param[in] n_in      Input n-direction-cosine values.
 * @param[out] n_out    Output n-direction-cosine values.
 * @param[in] a_in      Input Gaussian parameter a values.
 * @param[out] a_out    Output Gaussian parameter a values.
 * @param[in] b_in      Input Gaussian parameter b values.
 * @param[out] b_out    Output Gaussian parameter b values.
 * @param[in] c_in      Input Gaussian parameter c values.
 * @param[out] c_out    Output Gaussian parameter c values.
 * @param[in] maj_in    Input Gaussian major axis values.
 * @param[out] maj_out  Output Gaussian major axis values.
 * @param[in] min_in    Input Gaussian minor axis values.
 * @param[out] min_out  Output Gaussian minor axis values.
 * @param[in] pa_in     Input Gaussian position angle values.
 * @param[out] pa_out   Output Gaussian position angle values.
 */
OSKAR_EXPORT
void oskar_sky_copy_source_data_cuda_d(
        int num, int* num_out, const int* mask,
        const double* ra_in,   double* ra_out,
        const double* dec_in,  double* dec_out,
        const double* I_in,    double* I_out,
        const double* Q_in,    double* Q_out,
        const double* U_in,    double* U_out,
        const double* V_in,    double* V_out,
        const double* ref_in,  double* ref_out,
        const double* sp_in,   double* sp_out,
        const double* rm_in,   double* rm_out,
        const double* l_in,    double* l_out,
        const double* m_in,    double* m_out,
        const double* n_in,    double* n_out,
        const double* a_in,    double* a_out,
        const double* b_in,    double* b_out,
        const double* c_in,    double* c_out,
        const double* maj_in,  double* maj_out,
        const double* min_in,  double* min_out,
        const double* pa_in,   double* pa_out
        );

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SKY_COPY_SOURCE_DATA_CUDA_H_ */
