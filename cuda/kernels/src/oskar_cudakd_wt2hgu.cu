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

#include "cuda/kernels/oskar_cudakd_wt2hgu.h"
#include "math/core/phase.h"

// Shared memory pointer used by the kernel.
extern __shared__ double smem[];

__global__
void oskar_cudakd_wt2hgu(const int na, const double* ax, const double* ay,
        const int nb, const double3* trig, const double k, double2* weights)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int a = blockDim.x * blockIdx.x + tx; // Antenna index.
    const int b = blockDim.y * blockIdx.y + ty; // Beam index.

    // Cache antenna and beam data from global memory,
    // avoiding shared memory bank conflicts.
    double* cax = smem;
    double* cay = &cax[blockDim.x];
    double* cbz = &cay[blockDim.x];
    double* cby = &cbz[blockDim.y];
    double* cbx = &cby[blockDim.y];
    if (a < na) {
        cax[tx] = ax[a];
        cay[tx] = ay[a];
    }
    if (b < nb) {
        cbx[ty] = trig[b].x;
        cby[ty] = trig[b].y;
        cbz[ty] = trig[b].z;
    }
    __syncthreads();

    // Compute the geometric phase of the beam direction.
    double2 weight;
    const double phase = -GEOMETRIC_PHASE_2D_HORIZONTAL(cax[tx], cay[tx],
            cbz[ty], cby[ty], cbx[ty], k);
    sincos(phase, &weight.y, &weight.x);
    // Do NOT normalise.

    // Write result to global memory.
    if (a < na && b < nb) {
        const int w = a + na * b;
        weights[w] = weight;
    }
}
