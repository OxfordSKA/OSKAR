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

#ifdef __cplusplus
extern "C" {
#endif

#include "oskar_windows.h"

struct oskar_SkyModelGlobal_f
{
    int num_sources;
    float* RA;          ///< The source Right Ascensions in radians.
    float* Dec;         ///< The source Declinations in radians.
    float* I;           ///< The source equatorial Stokes I values.
    float* Q;           ///< The source equatorial Stokes Q values.
    float* U;           ///< The source equatorial Stokes U values.
    float* V;           ///< The source equatorial Stokes V values.
};
typedef struct oskar_SkyModelGlobal_f oskar_SkyModelGlobal_f;

struct oskar_SkyModelLocal_f
{
    int num_sources;
    float* RA;          ///< The source Right Ascensions in radians.
    float* Dec;         ///< The source Declinations in radians.
    float* I;           ///< The source horizontal Stokes I values.
    float* Q;           ///< The source horizontal Stokes Q values.
    float* U;           ///< The source horizontal Stokes U values.
    float* V;           ///< The source horizontal Stokes V values.
    float* hor_l;       ///< The source horizontal l-direction-cosines.
    float* hor_m;       ///< The source horizontal m-direction-cosines.
    float* hor_n;       ///< The source horizontal n-direction-cosines.
};
typedef struct oskar_SkyModelLocal_f oskar_SkyModelLocal_f;

struct oskar_SkyModelGlobal_d
{
    int num_sources;
    double* RA;         ///< The source Right Ascensions in radians.
    double* Dec;        ///< The source Declinations in radians.
    double* I;          ///< The source equatorial Stokes I values.
    double* Q;          ///< The source equatorial Stokes Q values.
    double* U;          ///< The source equatorial Stokes U values.
    double* V;          ///< The source equatorial Stokes V values.
};
typedef struct oskar_SkyModelGlobal_d oskar_SkyModelGlobal_d;

struct oskar_SkyModelLocal_d
{
    int num_sources;
    double* RA;         ///< The source Right Ascensions in radians.
    double* Dec;        ///< The source Declinations in radians.
    double* I;          ///< The source horizontal Stokes I values.
    double* Q;          ///< The source horizontal Stokes Q values.
    double* U;          ///< The source horizontal Stokes U values.
    double* V;          ///< The source horizontal Stokes V values.
    double* hor_l;      ///< The source horizontal l-direction-cosines.
    double* hor_m;      ///< The source horizontal m-direction-cosines.
    double* hor_n;      ///< The source horizontal n-direction-cosines.
};
typedef struct oskar_SkyModelLocal_d oskar_SkyModelLocal_d;

#ifdef __cplusplus
}
#endif

#endif // OSKAR_SKY_MODEL_H_
