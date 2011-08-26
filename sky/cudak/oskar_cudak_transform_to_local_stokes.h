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

#ifndef OSKAR_CUDAK_TRANSFORM_TO_LOCAL_STOKES_H_
#define OSKAR_CUDAK_TRANSFORM_TO_LOCAL_STOKES_H_

/**
 * @file oskar_cudak_transform_to_local_stokes.h
 */

#include "utility/oskar_cuda_eclipse.h"

/**
 * @brief
 * CUDA kernel to transform Stokes Q and Stokes U from the equatorial to the
 * horizontal frame (single precision).
 *
 * @details
 * This CUDA kernel transforms equatorial Stokes parameters to local
 * (horizontal) Stokes parameters by rotating (mixing) Stokes Q and Stokes U.
 *
 * @param[in] ns           The number of source positions.
 * @param[in] ra           The source Right Ascensions in radians.
 * @param[in] dec          The source Declinations in radians.
 * @param[in] cos_lat      The cosine of the geographic latitude.
 * @param[in] sin_lat      The sine of the geographic latitude.
 * @param[in] lst          The local sidereal time in radians.
 * @param[in,out] stokes_Q The original and transformed Stokes Q values.
 * @param[in,out] stokes_U The original and transformed Stokes U values.
 */
__global__
void oskar_cudak_transform_to_local_stokes_f(int ns, const float* ra,
        const float* dec, float cos_lat, float sin_lat, float lst,
        float* stokes_Q, float* stokes_U);

/**
 * @brief
 * CUDA kernel to transform Stokes Q and Stokes U from the equatorial to the
 * horizontal frame (double precision).
 *
 * @details
 * This CUDA kernel transforms equatorial Stokes parameters to local
 * (horizontal) Stokes parameters by rotating (mixing) Stokes Q and Stokes U.
 *
 * @param[in] ns           The number of source positions.
 * @param[in] ra           The source Right Ascensions in radians.
 * @param[in] dec          The source Declinations in radians.
 * @param[in] cos_lat      The cosine of the geographic latitude.
 * @param[in] sin_lat      The sine of the geographic latitude.
 * @param[in] lst          The local sidereal time in radians.
 * @param[in,out] stokes_Q The original and transformed Stokes Q values.
 * @param[in,out] stokes_U The original and transformed Stokes U values.
 */
__global__
void oskar_cudak_transform_to_local_stokes_d(int ns, const double* ra,
        const double* dec, double cos_lat, double sin_lat, double lst,
        double* stokes_Q, double* stokes_U);

#endif // OSKAR_CUDAK_TRANSFORM_TO_LOCAL_STOKES_H_
