/*
 * Copyright (c) 2012, The University of Oxford
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

#include "interferometry/oskar_evaluate_jones_K_cuda.h"
#include "math/cudak/oskar_cudak_dftw_3d_seq_out.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_evaluate_jones_K_cuda_f(float2* d_jones, int num_stations,
        const float* d_u, const float* d_v, const float* d_w, int num_sources,
        const float* d_l, const float* d_m, const float* d_n)
{
    /* Define block and grid sizes. */
    const dim3 num_threads(64, 4); /* Sources, stations. */
    const dim3 num_blocks((num_sources + num_threads.x - 1) / num_threads.x,
            (num_stations + num_threads.y - 1) / num_threads.y);
    const size_t s_mem = 3 * (num_threads.x + num_threads.y) * sizeof(float);

    /* Compute DFT phase weights for K. */
    oskar_cudak_dftw_3d_seq_out_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads, s_mem)
    (num_stations, d_u, d_v, d_w, num_sources, d_l, d_m, d_n, d_jones);
}

/* Double precision. */
void oskar_evaluate_jones_K_cuda_d(double2* d_jones, int num_stations,
        const double* d_u, const double* d_v, const double* d_w, int num_sources,
        const double* d_l, const double* d_m, const double* d_n)
{
    /* Define block and grid sizes. */
    const dim3 num_threads(64, 4); /* Sources, stations. */
    const dim3 num_blocks((num_sources + num_threads.x - 1) / num_threads.x,
            (num_stations + num_threads.y - 1) / num_threads.y);
    const size_t s_mem = 3 * (num_threads.x + num_threads.y) * sizeof(double);

    /* Compute DFT phase weights for K. */
    oskar_cudak_dftw_3d_seq_out_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads, s_mem)
    (num_stations, d_u, d_v, d_w, num_sources, d_l, d_m, d_n, d_jones);
}

#ifdef __cplusplus
}
#endif


/* Kernels. ================================================================ */
