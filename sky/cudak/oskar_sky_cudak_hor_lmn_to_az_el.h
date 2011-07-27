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

#ifndef OSKAR_SKY_CUDAK_HOR_LMN_TO_AZ_EL_H_
#define OSKAR_SKY_CUDAK_HOR_LMN_TO_AZ_EL_H_

/**
 * @file oskar_sky_cudak_hor_lmn_to_az_el.h
 */

#include "utility/oskar_util_cuda_eclipse.h"

/**
 * @brief
 * CUDA kernel to compute azimuth and elevation from horizontal direction
 * cosines (single precision).
 *
 * @details
 * This CUDA kernel computes the azimuth and elevation from the given
 * horizontal direction cosines.
 *
 * The directions are:
 * <li> l is parallel to x (pointing East), </li>
 * <li> m is parallel to y (pointing North), </li>
 * <li> n is parallel to z (pointing to the zenith). </li>
 *
 * @param[in] n    The number of points.
 * @param[in] p_l  The l-direction-cosines.
 * @param[in] p_m  The m-direction-cosines.
 * @param[in] p_n  The n-direction-cosines.
 * @param[out] az  The azimuths, in radians.
 * @param[out] el  The elevations, in radians.
 */
__global__
void oskar_sky_cudakf_hor_lmn_to_az_el(int n, const float* p_l,
        const float* p_m, const float* p_n, float* az, float* el);

/**
 * @brief
 * CUDA kernel to compute azimuth and elevation from horizontal direction
 * cosines (double precision).
 *
 * @details
 * This CUDA kernel computes the azimuth and elevation from the given
 * horizontal direction cosines.
 *
 * The directions are:
 * <li> l is parallel to x (pointing East), </li>
 * <li> m is parallel to y (pointing North), </li>
 * <li> n is parallel to z (pointing to the zenith). </li>
 *
 * @param[in] n    The number of points.
 * @param[in] p_l  The l-direction-cosines.
 * @param[in] p_m  The m-direction-cosines.
 * @param[in] p_n  The n-direction-cosines.
 * @param[out] az  The azimuths, in radians.
 * @param[out] el  The elevations, in radians.
 */
__global__
void oskar_sky_cudakd_hor_lmn_to_az_el(int n, const double* p_l,
        const double* p_m, const double* p_n, double* az, double* el);

#endif // OSKAR_SKY_CUDAK_HOR_LMN_TO_AZ_EL_H_
