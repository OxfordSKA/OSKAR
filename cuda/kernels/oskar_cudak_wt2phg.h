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

#ifndef OSKAR_CUDAK_WT2PHG_H_
#define OSKAR_CUDAK_WT2PHG_H_

/**
 * @file oskar_cudak_wt2phg.h
 */

#include "cuda/CudaEclipse.h"

/**
 * @brief
 * CUDA kernel to generate un-normalised geometric beamforming weights
 * (single precision).
 *
 * @details
 * This CUDA kernel produces the complex antenna beamforming weights for the
 * given directions, and stores them in device memory.
 * The weights are NOT normalised by the number of antennas.
 *
 * Each thread generates the complex weights for a single antenna and a single
 * beam direction.
 *
 * The kernel requires (blockDim.x + blockDim.y) * sizeof(double2)
 * bytes of shared memory.
 *
 * @param[in] na Number of antennas.
 * @param[in] ax Array of antenna x positions in wavenumbers.
 * @param[in] ay Array of antenna y positions in wavenumbers.
 * @param[in] nb Number of beams.
 * @param[in] bcace The beam (cosine azimuth * cosine elevation).
 * @param[in] bsace The beam (sine azimuth * cosine elevation).
 * @param[out] weights Matrix of complex antenna weights (na columns, nb rows).
 */
__global__
void oskar_cudakf_wt2phg(const int na, const float* ax, const float* ay,
        const int nb, const float* bcace, const float* bsace, float2* weights);

/**
 * @brief
 * CUDA kernel to generate un-normalised geometric beamforming weights
 * (double precision).
 *
 * @details
 * This CUDA kernel produces the complex antenna beamforming weights for the
 * given directions, and stores them in device memory.
 * The weights are NOT normalised by the number of antennas.
 *
 * Each thread generates the complex weights for a single antenna and a single
 * beam direction.
 *
 * The kernel requires (blockDim.x + blockDim.y) * sizeof(double2)
 * bytes of shared memory.
 *
 * @param[in] na Number of antennas.
 * @param[in] ax Array of antenna x positions in metres.
 * @param[in] ay Array of antenna y positions in metres.
 * @param[in] nb Number of beams.
 * @param[in] bcace The beam (cosine azimuth * cosine elevation).
 * @param[in] bsace The beam (sine azimuth * cosine elevation).
 * @param[out] weights Matrix of complex antenna weights (na columns, nb rows).
 */
__global__
void oskar_cudakd_wt2phg(const int na, const double* ax, const double* ay,
        const int nb, const double* bcace, const double* bsace,
        double2* weights);

#endif // OSKAR_CUDAK_WT2PHG_H_
