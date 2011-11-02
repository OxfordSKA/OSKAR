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

#ifndef OSKAR_XYZ_TO_UVW_CUDA_H_
#define OSKAR_XYZ_TO_UVW_CUDA_H_

/**
 * @file oskar_xyz_to_uvw_cuda.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Transforms station (x,y,z) coordinates to (u,v,w) coordinates
 * (single precision).
 *
 * @details
 * Given the hour angle and declination of the phase tracking centre, this
 * function transforms the station (x,y,z) coordinates to (u,v,w) coordinates.
 *
 * @param[in]  n    The number of antennas.
 * @param[in]  d_x  The station x-positions (length n).
 * @param[in]  d_y  The station y-positions (length n).
 * @param[in]  d_z  The station z-positions (length n).
 * @param[in]  ha0  The hour angle of the phase tracking centre in radians.
 * @param[in]  dec0 The declination of the phase tracking centre in radians.
 * @param[out] d_u  The station u-positions (length n).
 * @param[out] d_v  The station v-positions (length n).
 * @param[out] d_w  The station w-positions (length n).
 */
OSKAR_EXPORT
int oskar_xyz_to_uvw_cuda_f(int n, const float* d_x, const float* d_y,
        const float* d_z, float ha0, float dec0, float* d_u, float* d_v,
        float* d_w);

/**
 * @brief
 * Transforms station (x,y,z) coordinates to (u,v,w) coordinates
 * (double precision).
 *
 * @details
 * Given the hour angle and declination of the phase tracking centre, this
 * function transforms the station (x,y,z) coordinates to (u,v,w) coordinates.
 *
 * @param[in]  n    The number of antennas.
 * @param[in]  d_x  The station x-positions (length n).
 * @param[in]  d_y  The station y-positions (length n).
 * @param[in]  d_z  The station z-positions (length n).
 * @param[in]  ha0  The hour angle of the phase tracking centre in radians.
 * @param[in]  dec0 The declination of the phase tracking centre in radians.
 * @param[out] d_u  The station u-positions (length n).
 * @param[out] d_v  The station v-positions (length n).
 * @param[out] d_w  The station w-positions (length n).
 */
OSKAR_EXPORT
int oskar_xyz_to_uvw_cuda_d(int n, const double* d_x, const double* d_y,
        const double* d_z, double ha0, double dec0, double* d_u, double* d_v,
        double* d_w);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_XYZ_TO_UVW_CUDA_H_ */
