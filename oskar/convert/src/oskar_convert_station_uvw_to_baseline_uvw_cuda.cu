/*
 * Copyright (c) 2013-2014, The University of Oxford
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

#include <oskar_convert_station_uvw_to_baseline_uvw_cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_convert_station_uvw_to_baseline_uvw_cuda_f(int num_stations,
        const float* d_u, const float* d_v, const float* d_w, float* d_uu,
        float* d_vv, float* d_ww)
{
    int num_threads = 32;
    oskar_convert_station_uvw_to_baseline_uvw_cudak_f
        OSKAR_CUDAK_CONF(num_stations, num_threads)
        (num_stations, d_u, d_v, d_w, d_uu, d_vv, d_ww);
}

/* Double precision. */
void oskar_convert_station_uvw_to_baseline_uvw_cuda_d(int num_stations,
        const double* d_u, const double* d_v, const double* d_w, double* d_uu,
        double* d_vv, double* d_ww)
{
    int num_threads = 32;
    oskar_convert_station_uvw_to_baseline_uvw_cudak_d
        OSKAR_CUDAK_CONF(num_stations, num_threads)
        (num_stations, d_u, d_v, d_w, d_uu, d_vv, d_ww);
}

/* Kernels. ================================================================ */

/* Single precision. */
__global__
void oskar_convert_station_uvw_to_baseline_uvw_cudak_f(int num_stations,
        const float* u, const float* v, const float* w, float* uu,
        float* vv, float* ww)
{
    /* Get first station index from block ID. */
    int s1 = blockIdx.x;

    /* Each thread does one baseline. */
    for (int s2 = s1 + threadIdx.x + 1; s2 < num_stations; s2 += blockDim.x)
    {
        /* Determine baseline index from station IDs. */
        int b = s1 * (num_stations - 1) - (s1 - 1) * s1/2 + s2 - s1 - 1;

        /* Compute baselines. */
        uu[b] = u[s2] - u[s1];
        vv[b] = v[s2] - v[s1];
        ww[b] = w[s2] - w[s1];
    }
}

/* Double precision. */
__global__
void oskar_convert_station_uvw_to_baseline_uvw_cudak_d(int num_stations,
        const double* u, const double* v, const double* w, double* uu,
        double* vv, double* ww)
{
    /* Get first station index from block ID. */
    int s1 = blockIdx.x;

    /* Each thread does one baseline. */
    for (int s2 = s1 + threadIdx.x + 1; s2 < num_stations; s2 += blockDim.x)
    {
        /* Determine baseline index from station IDs. */
        int b = s1 * (num_stations - 1) - (s1 - 1) * s1/2 + s2 - s1 - 1;

        /* Compute baselines. */
        uu[b] = u[s2] - u[s1];
        vv[b] = v[s2] - v[s1];
        ww[b] = w[s2] - w[s1];
    }
}

#ifdef __cplusplus
}
#endif
