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

#ifndef OSKAR_INTERFEROMETRY_CUDAK_XYZ2UVW_H_
#define OSKAR_INTERFEROMETRY_CUDAK_XYZ2UVW_H_

/**
 * @file oskar_cudak_xyz2uvw.h
 */

#include "oskar_global.h"

/**
 * @brief
 * CUDA kernel to rotate station (x,y,z) to (u,v,w) coordinates
 * (single precision).
 *
 * @details
 * CUDA kernel to rotate station (x,y,z) to (u,v,w) coordinates
 * (single precision).
 *
 * @param[in] n    Number of stations.
 * @param[in] x    Station x positions.
 * @param[in] x    Station y positions.
 * @param[in] z    Station z positions.
 * @param[in] ha0  Hour angle of phase centre (radians).
 * @param[in] dec0 Declination of phase centre (radians).
 * @param[out] u   Station u positions.
 * @param[out] v   Station v positions.
 * @param[out] w   Station w positions.
 */
__global__
void oskar_cudak_xyz_to_uvw_f(int n, const float* x, const float* y,
        const float* z, float ha0, float dec0, float* u, float* v, float* w);

/**
 * @brief
 * CUDA kernel to rotate station (x,y,z) to (u,v,w) coordinates
 * (double precision).
 *
 * @details
 * CUDA kernel to rotate station (x,y,z) to (u,v,w) coordinates
 * (double precision).
 *
 * @param[in] n    Number of stations.
 * @param[in] x    Station x positions.
 * @param[in] x    Station y positions.
 * @param[in] z    Station z positions.
 * @param[in] ha0  Hour angle of phase centre (radians).
 * @param[in] dec0 Declination of phase centre (radians).
 * @param[out] u   Station u positions.
 * @param[out] v   Station v positions.
 * @param[out] w   Station w positions.
 */
__global__
void oskar_cudak_xyz_to_uvw_d(int n, const double* x, const double* y,
        const double* z, double ha0, double dec0, double* u, double* v,
        double* w);

#endif // OSKAR_INTERFEROMETRY_CUDAK_XYZ2UVW_H_
