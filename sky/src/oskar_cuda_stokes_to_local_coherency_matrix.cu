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

#include "sky/oskar_cuda_stokes_to_local_coherency_matrix.h"
#include "sky/cudak/oskar_cudak_stokes_to_local_coherency_matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

// Single precision.

int oskar_cuda_stokes_to_local_coherency_matrix_f(float lst, float lat,
		oskar_SkyModelLocal_f* hd_sky)
{
	// Precompute latitude trigonometry.
	float cos_lat = cosf(lat);
	float sin_lat = sinf(lat);

	// Set up thread and block dimensions.
	const int n = hd_sky->num_sources;
    const int n_thd = 256;
    const int n_blk = (n + n_thd - 1) / n_thd;

    // Compute the local coherency matrix.
	oskar_cudak_stokes_to_local_coherency_matrix_f <<< n_blk, n_thd >>> (n,
			hd_sky->RA, hd_sky->Dec, hd_sky->I, hd_sky->Q, hd_sky->U,
			hd_sky->V, cos_lat, sin_lat, lst, hd_sky->B);
    cudaDeviceSynchronize();

	return cudaPeekAtLastError();
}

// Double precision.

int oskar_cuda_stokes_to_local_coherency_matrix_d(double lst, double lat,
		oskar_SkyModelLocal_d* hd_sky)
{
	// Precompute latitude trigonometry.
	double cos_lat = cos(lat);
	double sin_lat = sin(lat);

	// Set up thread and block dimensions.
	const int n = hd_sky->num_sources;
    const int n_thd = 256;
    const int n_blk = (n + n_thd - 1) / n_thd;

    // Compute the local coherency matrix.
	oskar_cudak_stokes_to_local_coherency_matrix_d <<< n_blk, n_thd >>> (n,
			hd_sky->RA, hd_sky->Dec, hd_sky->I, hd_sky->Q, hd_sky->U,
			hd_sky->V, cos_lat, sin_lat, lst, hd_sky->B);
    cudaDeviceSynchronize();

	return cudaPeekAtLastError();
}

#ifdef __cplusplus
}
#endif
