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

#ifndef OSKAR_SKY_CUDA_HORIZON_CLIP_H_
#define OSKAR_SKY_CUDA_HORIZON_CLIP_H_

/**
 * @file oskar_sky_cuda_horizon_clip.h
 */

#include "oskar_sky_windows.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Clips sources below the horizon (single precision).
 *
 * @details
 * This function determines which sources are above the horizon, and fills
 * arrays containing the source coordinates and brightnesses for those
 * sources.
 *
 * All arrays (including the output arrays) should be of length n_in, but
 * the number of sources above the horizon is returned in n_out, and only the
 * first n_out elements of the output arrays should be used.
 *
 * @param[in] n_in   The number of input sources in the sky model.
 * @param[in] in_b   The input source brightnesses.
 * @param[in] ra     The input source Right Ascensions in radians.
 * @param[in] dec    The input source Declinations in radians.
 * @param[in] ra0    The Right Ascension of the phase centre.
 * @param[in] dec0   The Declination of the phase centre.
 * @param[in] lst    The current local sidereal time in radians.
 * @param[in] lat    The geographic latitude of the observer.
 * @param[out] n_out The number of sources above the horizon.
 * @param[out] out_b The output source brightnesses.
 * @param[out] eq_l  The source l-direction-cosines in the equatorial system.
 * @param[out] eq_m  The source m-direction-cosines in the equatorial system.
 * @param[out] hor_l The source l-direction-cosines in the horizontal system.
 * @param[out] hor_m The source m-direction-cosines in the horizontal system.
 * @param[in] work   Work array of length n_in.
 */
DllExport
int oskar_sky_cudaf_horizon_clip(int n_in, const float* in_b,
        const float* ra, const float* dec, float ra0, float dec0,
        float lst, float lat, int* n_out, float* out_b, float* eq_l,
        float* eq_m, float* hor_l, float* hor_m, float* work);

/**
 * @brief
 * Clips sources below the horizon (double precision).
 *
 * @details
 * This function determines which sources are above the horizon, and fills
 * arrays containing the source coordinates and brightnesses for those
 * sources.
 *
 * All arrays (including the output arrays) should be of length n_in, but
 * the number of sources above the horizon is returned in n_out, and only the
 * first n_out elements of the output arrays should be used.
 *
 * @param[in] n_in   The number of input sources in the sky model.
 * @param[in] in_b   The input source brightnesses.
 * @param[in] ra     The input source Right Ascensions in radians.
 * @param[in] dec    The input source Declinations in radians.
 * @param[in] ra0    The Right Ascension of the phase centre.
 * @param[in] dec0   The Declination of the phase centre.
 * @param[in] lst    The current local sidereal time in radians.
 * @param[in] lat    The geographic latitude of the observer.
 * @param[out] n_out The number of sources above the horizon.
 * @param[out] out_b The output source brightnesses.
 * @param[out] eq_l  The source l-direction-cosines in the equatorial system.
 * @param[out] eq_m  The source m-direction-cosines in the equatorial system.
 * @param[out] hor_l The source l-direction-cosines in the horizontal system.
 * @param[out] hor_m The source m-direction-cosines in the horizontal system.
 * @param[in] work   Work array of length n_in.
 */
DllExport
int oskar_sky_cudad_horizon_clip(int n_in, const double* in_b,
        const double* ra, const double* dec, double lst, double lat,
        int* n_out, double* out_b, double* eq_l, double* eq_m,
        double* hor_l, double* hor_m);

#ifdef __cplusplus
}
#endif

#endif // OSKAR_SKY_CUDA_HORIZON_CLIP_H_
