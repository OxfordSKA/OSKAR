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

#include "interferometry/oskar_cuda_xyz_to_uvw.h"
#include "interferometry/cudak/oskar_cudak_xyz_to_uvw.h"

#ifdef __cplusplus
extern "C" {
#endif

// Single precision.
int oskar_cuda_xyz_to_uvw_f(int n, const float* d_x, const float* d_y,
		const float* d_z, float ha0, float dec0, float* d_u, float* d_v,
		float* d_w)
{
    // Define block and grid sizes.
    const int n_thd = 256;
    const int n_blk = (n + n_thd - 1) / n_thd;

	// Call the CUDA kernel.
    oskar_cudak_xyz_to_uvw_f OSKAR_CUDAK_CONF(n_blk, n_thd)
    (n, d_x, d_y, d_z, ha0, dec0, d_u, d_v, d_w);
    cudaDeviceSynchronize();
    return cudaPeekAtLastError();
}

// Double precision.
int oskar_cuda_xyz_to_uvw_d(int n, const double* d_x, const double* d_y,
        const double* d_z, double ha0, double dec0, double* d_u, double* d_v,
        double* d_w)
{
    // Define block and grid sizes.
    const int n_thd = 256;
    const int n_blk = (n + n_thd - 1) / n_thd;

	// Call the CUDA kernel.
    oskar_cudak_xyz_to_uvw_d OSKAR_CUDAK_CONF(n_blk, n_thd)
    (n, d_x, d_y, d_z, ha0, dec0, d_u, d_v, d_w);
    cudaDeviceSynchronize();
    return cudaPeekAtLastError();
}

#ifdef __cplusplus
}
#endif
