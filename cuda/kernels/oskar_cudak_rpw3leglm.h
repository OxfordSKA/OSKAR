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

#ifndef OSKAR_CUDAK_RPW3LEGLM_H_
#define OSKAR_CUDAK_RPW3LEGLM_H_

/**
 * @file oskar_cudak_rpw3leglm.h
 */

#include "cuda/CudaEclipse.h"

/**
 * @brief
 * CUDA kernel to compute relative geometric phases of specified sources.
 *
 * @details
 *
 * @param[in] na Number of antennas.
 * @param[in] ax Array of antenna x positions in metres (length na).
 * @param[in] ay Array of antenna y positions in metres (length na).
 * @param[in] az Array of antenna z positions in metres (length na).
 * @param[in] scha0 The sine and cosine of the phase reference Hour Angle.
 * @param[in] scdec0 The sine and cosine of the phase reference Declination.
 * @param[in] ns The number of source positions.
 * @param[in] ha The source Hour Angle coordinates in radians (length ns).
 * @param[in] dec The source Declination coordinates in radians (length ns).
 * @param[in] k The wavenumber (rad / m).
 * @param[out] phases The computed phases (see note, above).
 */
__global__
void oskar_cudak_rpw3leglm(const int na, const float* ax, const float* ay,
        const float* az, const float2 scha0, const float2 scdec0, const int ns,
        const float* ha, const float* dec, const float k, float2* weights);

#endif // OSKAR_CUDAK_RPW3LEGLM_H_
