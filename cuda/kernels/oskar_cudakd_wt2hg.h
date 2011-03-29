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

#ifndef OSKAR_CUDAKD_WT2HG_H_
#define OSKAR_CUDAKD_WT2HG_H_

/**
 * @file oskar_cudakd_wt2hg.h
 */

#include "cuda/CudaEclipse.h"

/**
 * @brief
 * CUDA kernel to generate beamforming weights.
 *
 * @details
 * This CUDA kernel produces the complex antenna beamforming weights for the
 * given directions, and stores them in device memory.
 * Each thread generates the complex weights for a single antenna and a single
 * beam direction.
 *
 * The input \p trig array contains triplets of the following pre-computed
 * trigonometry:
 *
 * trig.x = {cosine azimuth}
 * trig.y = {sine azimuth}
 * trig.z = {cosine elevation}
 *
 * The kernel requires blockDim.x * sizeof(double2) +
 * blockDim.y * sizeof(double3) bytes of shared memory.
 *
 * The number of doubleing-point operations performed by this kernel is:
 * \li Sines and cosines: 2 * na * nb.
 * \li Multiplies: 4 * na * nb.
 * \li Divides: 2 * na * nb.
 * \li Additions / subtractions: na * nb.
 *
 * @param[in] na Number of antennas.
 * @param[in] ax Array of antenna x positions in metres.
 * @param[in] ay Array of antenna y positions in metres.
 * @param[in] nb Number of beams.
 * @param[in] trig Precomputed trigonometry for all beam positions.
 * @param[in] k The wavenumber (rad / m).
 * @param[out] weights Matrix of complex antenna weights (na columns, nb rows).
 */
__global__
void oskar_cudakd_wt2hg(const int na, const double* ax, const double* ay,
        const int nb, const double3* trig, const double k, double2* weights);

#endif // OSKAR_CUDAKD_WT2HG_H_
