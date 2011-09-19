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

#ifndef OSKAR_CUDA_TRANSFORM_TO_LOCAL_STOKES_H_
#define OSKAR_CUDA_TRANSFORM_TO_LOCAL_STOKES_H_

/**
 * @file oskar_cuda_transform_to_local_stokes.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Convert equatorial Stokes parameters to local Stokes parameters
 * (single precision).
 *
 * @details
 * This function converts the source Stokes parameters from the equatorial
 * frame to the local horizontal frame.
 *
 * @param[in] ns       The number of sources.
 * @param[in] d_ra     The source Right Ascension positions in radians.
 * @param[in] d_dec    The source Declination positions in radians.
 * @param[in] lst      The local sidereal time.
 * @param[in] lat      The geographic latitude.
 * @param[in,out] d_Q  The original and transformed Stokes Q parameters.
 * @param[in,out] d_U  The original and transformed Stokes U parameters.
 */
OSKAR_EXPORT
int oskar_cuda_transform_to_local_stokes_f(int ns, const float* d_ra,
        const float* d_dec, float lst, float lat, float* d_Q, float* d_U);

/**
 * @brief
 * Convert equatorial Stokes parameters to local Stokes parameters
 * (double precision).
 *
 * @details
 * This function converts the source Stokes parameters from the equatorial
 * frame to the local horizontal frame.
 *
 * @param[in] ns       The number of sources.
 * @param[in] d_ra     The source Right Ascension positions in radians.
 * @param[in] d_dec    The source Declination positions in radians.
 * @param[in] lst      The local sidereal time.
 * @param[in] lat      The geographic latitude.
 * @param[in,out] d_Q  The original and transformed Stokes Q parameters.
 * @param[in,out] d_U  The original and transformed Stokes U parameters.
 */
OSKAR_EXPORT
int oskar_cuda_transform_to_local_stokes_d(int ns, const double* d_ra,
        const double* d_dec, double lst, double lat, double* d_Q, double* d_U);

#ifdef __cplusplus
}
#endif

#endif // OSKAR_CUDA_TRANSFORM_TO_LOCAL_STOKES_H_
