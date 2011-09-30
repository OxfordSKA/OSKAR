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

#ifndef OSKAR_SKY_MODEL_H_
#define OSKAR_SKY_MODEL_H_

/**
 * @file oskar_SkyModel.h
 */

#include "oskar_global.h"


#ifdef __cplusplus
extern "C" {
#endif

struct oskar_SkyModelGlobal_f
{
    int num_sources;

    float* RA;             ///< The source Right Ascensions, in radians.
    float* Dec;            ///< The source Declinations, in radians.

    float* I;              ///< The source equatorial Stokes I values, in Jy.
    float* Q;              ///< The source equatorial Stokes Q values, in Jy.
    float* U;              ///< The source equatorial Stokes U values, in Jy.
    float* V;              ///< The source equatorial Stokes V values, in Jy.

    float* reference_freq; ///< The reference frequency for the source brightness, in Hz.
    float* spectral_index; ///< The source spectral index.

    float* rel_l;          ///< Source l-direction-cosines relative to the phase centre.
    float* rel_m;          ///< Source m-direction-cosines relative to the phase centre.
    float* rel_n;          ///< Source n-direction-cosines relative to the phase centre.
};
typedef struct oskar_SkyModelGlobal_f oskar_SkyModelGlobal_f;


struct oskar_SkyModelGlobal_d
{
    int num_sources;

    double* RA;             ///< The source Right Ascensions, in radians.
    double* Dec;            ///< The source Declinations, in radians.

    double* I;              ///< The source equatorial Stokes I values, in Jy.
    double* Q;              ///< The source equatorial Stokes Q values, in Jy.
    double* U;              ///< The source equatorial Stokes U values, in Jy.
    double* V;              ///< The source equatorial Stokes V values, in Jy.

    double* reference_freq; ///< The reference frequency for the source brightness, in Hz.
    double* spectral_index; ///< The source spectral index.

    double* rel_l;          ///< Source l-direction-cosines relative to the phase centre.
    double* rel_m;          ///< Source m-direction-cosines relative to the phase centre.
    double* rel_n;          ///< Source n-direction-cosines relative to the phase centre.
};
typedef struct oskar_SkyModelGlobal_d oskar_SkyModelGlobal_d;


struct oskar_SkyModelLocal_f
{
    int num_sources;

    float* RA;          ///< Source Right Ascensions, in radians.
    float* Dec;         ///< Source Declinations, in radians.

    float* I;           ///< Source horizontal Stokes I values, in Jy.
    float* Q;           ///< Source horizontal Stokes Q values, in Jy.
    float* U;           ///< Source horizontal Stokes U values, in Jy.
    float* V;           ///< Source horizontal Stokes V values, in Jy.

    float* hor_l;       ///< Source horizontal l-direction-cosines.
    float* hor_m;       ///< Source horizontal m-direction-cosines.
    float* hor_n;       ///< Source horizontal n-direction-cosines.

    float* rel_l;      ///< Source l-direction-cosines relative to the phase centre.
    float* rel_m;      ///< Source m-direction-cosines relative to the phase centre.
    float* rel_n;      ///< Source n-direction-cosines relative to the phase centre.
};
typedef struct oskar_SkyModelLocal_f oskar_SkyModelLocal_f;


struct oskar_SkyModelLocal_d
{
    int num_sources;

    double* RA;         ///< Source Right Ascensions in radians.
    double* Dec;        ///< Source Declinations in radians.

    double* I;          ///< Source horizontal Stokes I values, in Jy.
    double* Q;          ///< Source horizontal Stokes Q values, in Jy.
    double* U;          ///< Source horizontal Stokes U values, in Jy.
    double* V;          ///< Source horizontal Stokes V values, in Jy.

    double* hor_l;      ///< Source horizontal l-direction-cosines.
    double* hor_m;      ///< Source horizontal m-direction-cosines.
    double* hor_n;      ///< Source horizontal n-direction-cosines.

    double* rel_l;      ///< Source l-direction-cosines relative to the phase centre.
    double* rel_m;      ///< Source m-direction-cosines relative to the phase centre.
    double* rel_n;      ///< Source n-direction-cosines relative to the phase centre.
};
typedef struct oskar_SkyModelLocal_d oskar_SkyModelLocal_d;


// ---------- Utility functions ------------------------------------------------
OSKAR_EXPORT
void oskar_sky_model_global_copy_to_gpu_d(const oskar_SkyModelGlobal_d* h_sky,
        oskar_SkyModelGlobal_d* hd_sky);

OSKAR_EXPORT
void oskar_sky_model_global_copy_to_gpu_f(const oskar_SkyModelGlobal_f* h_sky,
        oskar_SkyModelGlobal_f* hd_sky);

OSKAR_EXPORT
void oskar_local_sky_model_allocate_gpu_d(const int num_sources,
        oskar_SkyModelLocal_d* hd_sky);

OSKAR_EXPORT
void oskar_local_sky_model_allocate_gpu_f(const int num_sources,
        oskar_SkyModelLocal_f* hd_sky);

OSKAR_EXPORT
void oskar_global_sky_model_free_gpu_d(oskar_SkyModelGlobal_d* hd_sky);

OSKAR_EXPORT
void oskar_global_sky_model_free_gpu_f(oskar_SkyModelGlobal_f* hd_sky);

OSKAR_EXPORT
void oskar_local_sky_model_free_gpu_d(oskar_SkyModelLocal_d* hd_sky);

OSKAR_EXPORT
void oskar_local_sky_model_free_gpu_f(oskar_SkyModelLocal_f* hd_sky);
// -----------------------------------------------------------------------------

#ifdef __cplusplus
}
#endif

#endif // OSKAR_SKY_MODEL_H_
