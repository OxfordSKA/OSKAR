/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#include <oskar_accumulate_baseline_visibility_for_source.h>
#include <oskar_correlate_point_time_smearing_cuda.h>
#include <oskar_sinc.h>

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_correlate_point_time_smearing_cuda_f(int num_sources,
        int num_stations, const float4c* d_jones,
        const float* d_source_I, const float* d_source_Q,
        const float* d_source_U, const float* d_source_V,
        const float* d_source_l, const float* d_source_m,
        const float* d_source_n, const float* d_station_u,
        const float* d_station_v, const float* d_station_x,
        const float* d_station_y, float inv_wavelength, float frac_bandwidth,
        float time_int_sec, float gha0_rad, float dec0_rad, float4c* d_vis)
{
    dim3 num_threads(128, 1);
    dim3 num_blocks(num_stations, num_stations);
    size_t shared_mem = num_threads.x * sizeof(float4c);
    oskar_correlate_point_time_smearing_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
    (num_sources, num_stations, d_jones, d_source_I, d_source_Q, d_source_U,
            d_source_V, d_source_l, d_source_m, d_source_n, d_station_u,
            d_station_v, d_station_x, d_station_y, inv_wavelength,
            frac_bandwidth, time_int_sec, gha0_rad, dec0_rad, d_vis);
}

/* Double precision. */
void oskar_correlate_point_time_smearing_cuda_d(int num_sources,
        int num_stations, const double4c* d_jones,
        const double* d_source_I, const double* d_source_Q,
        const double* d_source_U, const double* d_source_V,
        const double* d_source_l, const double* d_source_m,
        const double* d_source_n, const double* d_station_u,
        const double* d_station_v, const double* d_station_x,
        const double* d_station_y, double inv_wavelength, double frac_bandwidth,
        double time_int_sec, double gha0_rad, double dec0_rad, double4c* d_vis)
{
    dim3 num_threads(128, 1);
    dim3 num_blocks(num_stations, num_stations);
    size_t shared_mem = num_threads.x * sizeof(double4c);
    oskar_correlate_point_time_smearing_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
    (num_sources, num_stations, d_jones, d_source_I, d_source_Q, d_source_U,
            d_source_V, d_source_l, d_source_m, d_source_n, d_station_u,
            d_station_v, d_station_x, d_station_y, inv_wavelength,
            frac_bandwidth, time_int_sec, gha0_rad, dec0_rad, d_vis);
}

#ifdef __cplusplus
}
#endif


/* Kernels. ================================================================ */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

#define OMEGA_EARTH  7.272205217e-5  /* radians/sec */
#define OMEGA_EARTHf 7.272205217e-5f /* radians/sec */

/* Indices into the visibility/baseline matrix. */
#define SI blockIdx.x /* Column index. */
#define SJ blockIdx.y /* Row index. */

extern __shared__ float4c  smem_f[];
extern __shared__ double4c smem_d[];

/* Single precision. */
__global__
void oskar_correlate_point_time_smearing_cudak_f(const int num_sources,
        const int num_stations, const float4c* __restrict__ jones,
        const float* __restrict__ source_I,
        const float* __restrict__ source_Q,
        const float* __restrict__ source_U,
        const float* __restrict__ source_V,
        const float* __restrict__ source_l,
        const float* __restrict__ source_m,
        const float* __restrict__ source_n,
        const float* __restrict__ station_u,
        const float* __restrict__ station_v,
        const float* __restrict__ station_x,
        const float* __restrict__ station_y, const float inv_wavelength,
        const float frac_bandwidth, const float time_int_sec,
        const float gha0_rad, const float dec0_rad,
        float4c* __restrict__ vis)
{
    /* Local variables. */
    float4c sum;
    float l, m, n, rb, rt;
    int i;

    /* Common values per thread block. */
    __shared__ float uu, vv, du_dt, dv_dt, dw_dt;
    __shared__ const float4c* __restrict__ station_i;
    __shared__ const float4c* __restrict__ station_j;

    /* Return immediately if in the wrong half of the visibility matrix. */
    if (SJ >= SI) return;

    /* Use thread 0 to set up the block. */
    if (threadIdx.x == 0)
    {
        float factor;

        /* Baseline lengths. */
        factor = M_PIf * inv_wavelength;
        uu = (station_u[SI] - station_u[SJ]) * factor;
        vv = (station_v[SI] - station_v[SJ]) * factor;

        /* Modify the baseline distance to include the common components
         * of the bandwidth smearing term. */
        uu *= frac_bandwidth;
        vv *= frac_bandwidth;

        /* Compute the derivatives for time-average smearing. */
        {
            float xx, yy, rot_angle, temp;
            float sin_HA, cos_HA, sin_Dec, cos_Dec;
            sincosf(gha0_rad, &sin_HA, &cos_HA);
            sincosf(dec0_rad, &sin_Dec, &cos_Dec);
            xx = (station_x[SI] - station_x[SJ]) * factor;
            yy = (station_y[SI] - station_y[SJ]) * factor;
            rot_angle = OMEGA_EARTHf * time_int_sec;
            temp = (xx * sin_HA + yy * cos_HA) * rot_angle;
            du_dt = (xx * cos_HA - yy * sin_HA) * rot_angle;
            dv_dt = temp * sin_Dec;
            dw_dt = -temp * cos_Dec;
        }

        /* Get pointers to source vectors for both stations. */
        station_i = &jones[num_sources * SI];
        station_j = &jones[num_sources * SJ];
    }
    __syncthreads();

    /* Partial sum per thread. */
    sum.a = make_float2(0.0f, 0.0f);
    sum.b = make_float2(0.0f, 0.0f);
    sum.c = make_float2(0.0f, 0.0f);
    sum.d = make_float2(0.0f, 0.0f);

    /* Each thread loops over a subset of the sources. */
    for (i = threadIdx.x; i < num_sources; i += blockDim.x)
    {
        /* Get source direction cosines. */
        l = source_l[i];
        m = source_m[i];
        n = source_n[i];

        /* Compute bandwidth- and time-smearing terms. */
        rb = oskar_sinc_f(uu * l + vv * m);
        rt = oskar_sinc_f(du_dt * l + dv_dt * m + dw_dt * n);
        rb *= rt;

        /* Accumulate baseline visibility response for source. */
        oskar_accumulate_baseline_visibility_for_source_f(&sum, i,
                source_I, source_Q, source_U, source_V,
                station_i, station_j, rb);
    }

    /* Store partial sum for the thread in shared memory and synchronise. */
    smem_f[threadIdx.x] = sum;
    __syncthreads();

    /* Accumulate contents of shared memory. */
    if (threadIdx.x == 0)
    {
        /* Sum over all sources for this baseline. */
        sum.a = make_float2(0.0f, 0.0f);
        sum.b = make_float2(0.0f, 0.0f);
        sum.c = make_float2(0.0f, 0.0f);
        sum.d = make_float2(0.0f, 0.0f);
        for (i = 0; i < blockDim.x; ++i)
        {
            sum.a.x += smem_f[i].a.x;
            sum.a.y += smem_f[i].a.y;
            sum.b.x += smem_f[i].b.x;
            sum.b.y += smem_f[i].b.y;
            sum.c.x += smem_f[i].c.x;
            sum.c.y += smem_f[i].c.y;
            sum.d.x += smem_f[i].d.x;
            sum.d.y += smem_f[i].d.y;
        }

        /* Determine 1D visibility index for global memory store. */
        i = SJ*(num_stations-1) - (SJ-1)*SJ/2 + SI - SJ - 1;

        /* Add result of this thread block to the baseline visibility. */
        vis[i].a.x += sum.a.x;
        vis[i].a.y += sum.a.y;
        vis[i].b.x += sum.b.x;
        vis[i].b.y += sum.b.y;
        vis[i].c.x += sum.c.x;
        vis[i].c.y += sum.c.y;
        vis[i].d.x += sum.d.x;
        vis[i].d.y += sum.d.y;
    }
}

/* Double precision. */
__global__
void oskar_correlate_point_time_smearing_cudak_d(const int num_sources,
        const int num_stations, const double4c* __restrict__ jones,
        const double* __restrict__ source_I,
        const double* __restrict__ source_Q,
        const double* __restrict__ source_U,
        const double* __restrict__ source_V,
        const double* __restrict__ source_l,
        const double* __restrict__ source_m,
        const double* __restrict__ source_n,
        const double* __restrict__ station_u,
        const double* __restrict__ station_v,
        const double* __restrict__ station_x,
        const double* __restrict__ station_y, const double inv_wavelength,
        const double frac_bandwidth, const double time_int_sec,
        const double gha0_rad, const double dec0_rad,
        double4c* __restrict__ vis)
{
    /* Local variables. */
    double4c sum;
    double l, m, n, r1, r2;
    int i;

    /* Common values per thread block. */
    __shared__ double uu, vv, du_dt, dv_dt, dw_dt;
    __shared__ const double4c* __restrict__ station_i;
    __shared__ const double4c* __restrict__ station_j;

    /* Return immediately if in the wrong half of the visibility matrix. */
    if (SJ >= SI) return;

    /* Use thread 0 to set up the block. */
    if (threadIdx.x == 0)
    {
        double factor;

        /* Baseline lengths. */
        factor = M_PI * inv_wavelength;
        uu = (station_u[SI] - station_u[SJ]) * factor;
        vv = (station_v[SI] - station_v[SJ]) * factor;

        /* Modify the baseline distance to include the common components
         * of the bandwidth smearing term. */
        uu *= frac_bandwidth;
        vv *= frac_bandwidth;

        /* Compute the derivatives for time-average smearing. */
        {
            double xx, yy, rot_angle, temp;
            double sin_HA, cos_HA, sin_Dec, cos_Dec;
            sincos(gha0_rad, &sin_HA, &cos_HA);
            sincos(dec0_rad, &sin_Dec, &cos_Dec);
            xx = (station_x[SI] - station_x[SJ]) * factor;
            yy = (station_y[SI] - station_y[SJ]) * factor;
            rot_angle = OMEGA_EARTH * time_int_sec;
            temp = (xx * sin_HA + yy * cos_HA) * rot_angle;
            du_dt = (xx * cos_HA - yy * sin_HA) * rot_angle;
            dv_dt = temp * sin_Dec;
            dw_dt = -temp * cos_Dec;
        }

        /* Get pointers to source vectors for both stations. */
        station_i = &jones[num_sources * SI];
        station_j = &jones[num_sources * SJ];
    }
    __syncthreads();

    /* Partial sum per thread. */
    sum.a = make_double2(0.0, 0.0);
    sum.b = make_double2(0.0, 0.0);
    sum.c = make_double2(0.0, 0.0);
    sum.d = make_double2(0.0, 0.0);

    /* Each thread loops over a subset of the sources. */
    for (i = threadIdx.x; i < num_sources; i += blockDim.x)
    {
        /* Get source direction cosines. */
        l = source_l[i];
        m = source_m[i];
        n = source_n[i];

        /* Compute bandwidth- and time-smearing terms. */
        r1 = oskar_sinc_d(uu * l + vv * m);
        r2 = oskar_sinc_d(du_dt * l + dv_dt * m + dw_dt * n);
        r1 *= r2;

        /* Accumulate baseline visibility response for source. */
        oskar_accumulate_baseline_visibility_for_source_d(&sum, i,
                source_I, source_Q, source_U, source_V,
                station_i, station_j, r1);
    }

    /* Store partial sum for the thread in shared memory and synchronise. */
    smem_d[threadIdx.x] = sum;
    __syncthreads();

    /* Accumulate contents of shared memory. */
    if (threadIdx.x == 0)
    {
        /* Sum over all sources for this baseline. */
        sum.a = make_double2(0.0, 0.0);
        sum.b = make_double2(0.0, 0.0);
        sum.c = make_double2(0.0, 0.0);
        sum.d = make_double2(0.0, 0.0);
        for (i = 0; i < blockDim.x; ++i)
        {
            sum.a.x += smem_d[i].a.x;
            sum.a.y += smem_d[i].a.y;
            sum.b.x += smem_d[i].b.x;
            sum.b.y += smem_d[i].b.y;
            sum.c.x += smem_d[i].c.x;
            sum.c.y += smem_d[i].c.y;
            sum.d.x += smem_d[i].d.x;
            sum.d.y += smem_d[i].d.y;
        }

        /* Determine 1D visibility index for global memory store. */
        i = SJ*(num_stations-1) - (SJ-1)*SJ/2 + SI - SJ - 1;

        /* Add result of this thread block to the baseline visibility. */
        vis[i].a.x += sum.a.x;
        vis[i].a.y += sum.a.y;
        vis[i].b.x += sum.b.x;
        vis[i].b.y += sum.b.y;
        vis[i].c.x += sum.c.x;
        vis[i].c.y += sum.c.y;
        vis[i].d.x += sum.d.x;
        vis[i].d.y += sum.d.y;
    }
}
