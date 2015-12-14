/*
 * Copyright (c) 2013-2014, The University of Oxford
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

#ifndef OSKAR_SKY_FILTER_BY_FLUX_CUDA_H_
#define OSKAR_SKY_FILTER_BY_FLUX_CUDA_H_

/**
 * @file oskar_sky_filter_by_flux_cuda.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Removes sources outside a given flux range using CUDA (single precision).
 *
 * @details
 * This function removes sources from all arrays that lie outside a given
 * flux range specified by \p min_I and \p max_I
 *
 * @param[in] num        Number of input sources.
 * @param[out] num_out   Number of sources retained.
 * @param[in] min_I      Minimum Stokes I flux.
 * @param[in] max_I      Maximum Stokes I flux.
 * @param[in,out] ra     Right ascension coordinates.
 * @param[in,out] dec    Declination coordinates.
 * @param[in,out] I      Stokes I values.
 * @param[in,out] Q      Stokes Q values.
 * @param[in,out] U      Stokes U values.
 * @param[in,out] V      Stokes V values.
 * @param[in,out] ref    Reference frequency values.
 * @param[in,out] sp     Spectral index values.
 * @param[in,out] rm     Rotation measure values.
 * @param[in,out] l      l-direction-cosine values.
 * @param[in,out] m      m-direction-cosine values.
 * @param[in,out] n      n-direction-cosine values.
 * @param[in,out] a      Gaussian parameter a values.
 * @param[in,out] b      Gaussian parameter b values.
 * @param[in,out] c      Gaussian parameter c values.
 * @param[in,out] maj    Gaussian major axis values.
 * @param[in,out] min    Gaussian minor axis values.
 * @param[in,out] pa     Gaussian position angle values.
 */
OSKAR_EXPORT
void oskar_sky_filter_by_flux_cuda_f(int num, int* num_out, float min_I,
        float max_I, float* ra, float* dec, float* I, float* Q, float* U,
        float* V, float* ref, float* sp, float* rm, float* l, float* m,
        float* n, float* a, float* b, float* c, float* maj, float* min,
        float* pa);

/**
 * @brief
 * Removes sources outside a given flux range using CUDA (double precision).
 *
 * @details
 * This function removes sources from all arrays that lie outside a given
 * flux range specified by \p min_I and \p max_I
 *
 * @param[in] num        Number of input sources.
 * @param[out] num_out   Number of sources retained.
 * @param[in] min_I      Minimum Stokes I flux.
 * @param[in] max_I      Maximum Stokes I flux.
 * @param[in,out] ra     Right ascension coordinates.
 * @param[in,out] dec    Declination coordinates.
 * @param[in,out] I      Stokes I values.
 * @param[in,out] Q      Stokes Q values.
 * @param[in,out] U      Stokes U values.
 * @param[in,out] V      Stokes V values.
 * @param[in,out] ref    Reference frequency values.
 * @param[in,out] sp     Spectral index values.
 * @param[in,out] rm     Rotation measure values.
 * @param[in,out] l      l-direction-cosine values.
 * @param[in,out] m      m-direction-cosine values.
 * @param[in,out] n      n-direction-cosine values.
 * @param[in,out] a      Gaussian parameter a values.
 * @param[in,out] b      Gaussian parameter b values.
 * @param[in,out] c      Gaussian parameter c values.
 * @param[in,out] maj    Gaussian major axis values.
 * @param[in,out] min    Gaussian minor axis values.
 * @param[in,out] pa     Gaussian position angle values.
 */
OSKAR_EXPORT
void oskar_sky_filter_by_flux_cuda_d(int num, int* num_out, double min_I,
        double max_I, double* ra, double* dec, double* I, double* Q, double* U,
        double* V, double* ref, double* sp, double* rm, double* l, double* m,
        double* n, double* a, double* b, double* c, double* maj, double* min,
        double* pa);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SKY_FILTER_BY_FLUX_CUDA_H_ */
