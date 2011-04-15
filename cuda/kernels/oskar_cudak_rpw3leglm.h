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
 * The antenna u,v,w coordinates must be supplied in a matrix of size (3, na),
 * where all the u-coordinates are given first, then all v, then all w.
 *
 * The complex phase weights are returned in a matrix of size (ns, na).
 *
 * @param[in] na Number of antennas.
 * @param[in] uvw Antenna u,v,w coordinates (see note, above).
 * @param[in] ns The number of source positions.
 * @param[in] l The source l-coordinates (length ns).
 * @param[in] m The source m-coordinates (length ns).
 * @param[in] n The source n-coordinates (length ns).
 * @param[in] k The wavenumber (rad / m).
 * @param[out] weights The computed complex phase weights (see note, above).
 */
__global__
void oskar_cudak_rpw3leglm(const int na, const float* uvw, const int ns,
        const float* l, const float* m, const float* n, const float k,
        float2* weights);

#endif // OSKAR_CUDAK_RPW3LEGLM_H_
