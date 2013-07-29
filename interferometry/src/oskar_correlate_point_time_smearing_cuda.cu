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

#include "interferometry/oskar_accumulate_baseline_visibility_for_source.h"
#include "interferometry/oskar_correlate_point_time_smearing_cuda.h"
#include "math/oskar_sinc.h"
#include <math.h>

#define STATION_BLOCK_SIZE 8

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
        const float* d_station_y, float freq_hz, float bandwidth_hz,
        float time_int_sec, float gha0_rad, float dec0_rad, float4c* d_vis)
{
    dim3 num_threads(128, 1);
    dim3 num_blocks(num_stations, num_stations);
    size_t shared_mem = num_threads.x * sizeof(float4c);
    oskar_correlate_point_time_smearing_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
    (num_sources, num_stations, d_jones, d_source_I, d_source_Q, d_source_U,
            d_source_V, d_source_l, d_source_m, d_source_n, d_station_u,
            d_station_v, d_station_x, d_station_y, freq_hz, bandwidth_hz,
            time_int_sec, gha0_rad, dec0_rad, d_vis);
}

/* Single precision. */
void oskar_correlate_point_time_smearing_cuda_2_f(int num_sources,
        int num_stations, const float4c* d_jones,
        const float* d_source_I, const float* d_source_Q,
        const float* d_source_U, const float* d_source_V,
        const float* d_source_l, const float* d_source_m,
        const float* d_source_n, const float* d_station_u,
        const float* d_station_v, const float* d_station_x,
        const float* d_station_y, float freq_hz, float bandwidth_hz,
        float time_int_sec, float gha0_rad, float dec0_rad, float4c* d_vis)
{
    dim3 num_threads(128, 1);
    int block_size = STATION_BLOCK_SIZE;
    int max_station_blocks = (num_stations + block_size - 1) / block_size;
    dim3 num_blocks(num_stations, max_station_blocks);
    size_t shared_mem = num_threads.x * sizeof(float4c); /* acc. buffer */
    shared_mem += num_threads.x * sizeof(float4c);       /* Jones cache */
    shared_mem += block_size * sizeof(float4c);          /* vis baseline buffer */
    shared_mem += num_threads.x * sizeof(float) * 4;     /* Source brightness */
    shared_mem += num_threads.x * sizeof(float) * 3;     /* Source coords */

    oskar_correlate_point_time_smearing_cudak_2_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
    (num_sources, num_stations, d_jones, d_source_I, d_source_Q, d_source_U,
            d_source_V, d_source_l, d_source_m, d_source_n, d_station_u,
            d_station_v, d_station_x, d_station_y, freq_hz, bandwidth_hz,
            time_int_sec, gha0_rad, dec0_rad, d_vis);
}


/* Double precision. */
void oskar_correlate_point_time_smearing_cuda_d(int num_sources,
        int num_stations, const double4c* d_jones,
        const double* d_source_I, const double* d_source_Q,
        const double* d_source_U, const double* d_source_V,
        const double* d_source_l, const double* d_source_m,
        const double* d_source_n, const double* d_station_u,
        const double* d_station_v, const double* d_station_x,
        const double* d_station_y, double freq_hz, double bandwidth_hz,
        double time_int_sec, double gha0_rad, double dec0_rad, double4c* d_vis)
{
    dim3 num_threads(128, 1);
    dim3 num_blocks(num_stations, num_stations);
    size_t shared_mem = num_threads.x * sizeof(double4c);
    oskar_correlate_point_time_smearing_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
    (num_sources, num_stations, d_jones, d_source_I, d_source_Q, d_source_U,
            d_source_V, d_source_l, d_source_m, d_source_n, d_station_u,
            d_station_v, d_station_x, d_station_y, freq_hz, bandwidth_hz,
            time_int_sec, gha0_rad, dec0_rad, d_vis);
}

#ifdef __cplusplus
}
#endif


/* Kernels. ================================================================ */

#define OMEGA_EARTH  7.272205217e-5  /* radians/sec */
#define OMEGA_EARTHf 7.272205217e-5f /* radians/sec */

/* Indices into the visibility/baseline matrix. */
#define SI blockIdx.x /* Column index. */
#define SJ blockIdx.y /* Row index. */

extern __shared__ float4c  smem_f4c[];
extern __shared__ double4c smem_d4c[];

/* Single precision. */
__global__
void oskar_correlate_point_time_smearing_cudak_f(const int num_sources,
        const int num_stations, const float4c* jones, const float* source_I,
        const float* source_Q, const float* source_U, const float* source_V,
        const float* source_l, const float* source_m, const float* source_n,
        const float* station_u, const float* station_v,
        const float* station_x, const float* station_y, const float freq_hz,
        const float bandwidth_hz, const float time_int_sec,
        const float gha0_rad, const float dec0_rad, float4c* vis)
{
    /* Local variables. */
    float4c sum;
    float l, m, n, rb, rt;
    int i;

    /* Common values per thread block. */
    __device__ __shared__ float uu, vv, du_dt, dv_dt, dw_dt;
    __device__ __shared__ const float4c *station_i, *station_j;

    /* Return immediately if in the wrong half of the visibility matrix. */
    if (SJ >= SI) return;

    /* Use thread 0 to set up the block. */
    if (threadIdx.x == 0)
    {
        float temp;

        /* Baseline lengths. */
        uu = (station_u[SI] - station_u[SJ]) * 0.5f;
        vv = (station_v[SI] - station_v[SJ]) * 0.5f;

        /* Modify the baseline distance to include the common components
         * of the bandwidth smearing term. */
        temp = bandwidth_hz / freq_hz; /* Fractional bandwidth */
        uu *= temp;
        vv *= temp;

        /* Compute the derivatives for time-average smearing. */
        {
            float xx, yy, rot_angle;
            float sin_HA, cos_HA, sin_Dec, cos_Dec;
            sincosf(gha0_rad, &sin_HA, &cos_HA);
            sincosf(dec0_rad, &sin_Dec, &cos_Dec);
            xx = (station_x[SI] - station_x[SJ]) * 0.5f;
            yy = (station_y[SI] - station_y[SJ]) * 0.5f;
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
    smem_f4c[threadIdx.x] = sum;
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
            sum.a.x += smem_f4c[i].a.x;
            sum.a.y += smem_f4c[i].a.y;
            sum.b.x += smem_f4c[i].b.x;
            sum.b.y += smem_f4c[i].b.y;
            sum.c.x += smem_f4c[i].c.x;
            sum.c.y += smem_f4c[i].c.y;
            sum.d.x += smem_f4c[i].d.x;
            sum.d.y += smem_f4c[i].d.y;
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


/* Single precision. */
__global__
void oskar_correlate_point_time_smearing_cudak_2_f(const int num_sources,
        const int num_stations, const float4c* __restrict__ jones,
        const float* __restrict__ source_I,
        const float* __restrict__ source_Q,
        const float* __restrict__ source_U,
        const float* __restrict__ source_V,
        const float* __restrict__ source_l,
        const float* __restrict__ source_m, const float* __restrict__ source_n,
        const float* __restrict__ station_u, const float* __restrict__ station_v,
        const float* __restrict__ station_x,
        const float* __restrict__ station_y, const float freq_hz,
        const float bandwidth_hz, const float time_int_sec,
        const float gha0_rad, const float dec0_rad, float4c* vis)
{
    int sJ0 = blockIdx.x+1;

    /* Return immediately if the whole block is outside the visibility matrix */
    /* i.e. there is nothing to do! */
    if (sJ0 + (blockIdx.y * STATION_BLOCK_SIZE) > num_stations)
        return;

    /* Pointers into dynamic shared memory */
    float4c* __restrict__ stationI   = smem_f4c + 0;
    float4c* __restrict__ Vpq_source = smem_f4c + blockDim.x;
    float4c* __restrict__ Vpq_sum    = smem_f4c + (blockDim.x*2);
    float* __restrict__ sourceI = (float*)(smem_f4c + (blockDim.x*2 + STATION_BLOCK_SIZE));
    float* __restrict__ sourceQ = sourceI + blockDim.x;
    float* __restrict__ sourceU = sourceQ + blockDim.x;
    float* __restrict__ sourceV = sourceU + blockDim.x;
    float* __restrict__ sourceL = sourceV + blockDim.x;
    float* __restrict__ sourceM = sourceL + blockDim.x;
    float* __restrict__ sourceN = sourceM + blockDim.x;

    /* Per baseline variables */
    __device__ __shared__ float uu[STATION_BLOCK_SIZE];
    __device__ __shared__ float vv[STATION_BLOCK_SIZE];
    __device__ __shared__ float du_dt[STATION_BLOCK_SIZE];
    __device__ __shared__ float dv_dt[STATION_BLOCK_SIZE];
    __device__ __shared__ float dw_dt[STATION_BLOCK_SIZE];
    if (threadIdx.x < STATION_BLOCK_SIZE)
    {
        /* Station index J of baseline I,J */
        int sJ = sJ0 + (blockIdx.y * STATION_BLOCK_SIZE) + threadIdx.x;

        /* Initialise per baseline vis */
        Vpq_sum[threadIdx.x].a = make_float2(0.0f, 0.0f);
        Vpq_sum[threadIdx.x].b = make_float2(0.0f, 0.0f);
        Vpq_sum[threadIdx.x].c = make_float2(0.0f, 0.0f);
        Vpq_sum[threadIdx.x].d = make_float2(0.0f, 0.0f);

        if (sJ < num_stations)
        {
            /* Baseline distances, in wavelengths. */
            float fractional_bandwidth = bandwidth_hz / freq_hz;
            uu[threadIdx.x] = (station_u[blockIdx.x] - station_u[sJ]) * 0.5f * fractional_bandwidth;
            vv[threadIdx.x] = (station_v[blockIdx.x] - station_v[sJ]) * 0.5f * fractional_bandwidth;
            float xx = (station_x[blockIdx.x] - station_x[sJ]) * 0.5f;
            float yy = (station_y[blockIdx.x] - station_y[sJ]) * 0.5f;

            /* Compute the derivatives for time-average smearing. */
            float rot_angle = OMEGA_EARTHf * time_int_sec;
            float sin_HA = sinf(gha0_rad);
            float cos_HA = cosf(gha0_rad);
            float sin_Dec = sinf(dec0_rad);
            float cos_Dec = cosf(dec0_rad);
            float temp = (xx * sin_HA + yy * cos_HA) * rot_angle;
            du_dt[threadIdx.x] = (xx * cos_HA - yy * sin_HA) * rot_angle;
            dv_dt[threadIdx.x] = temp * sin_Dec;
            dw_dt[threadIdx.x] = -temp * cos_Dec;
        }
    }
    __syncthreads();


    /* Loop in blocks of threadDim.x sources over each station pair and
     * accumulate forming visibilities. */
    int num_source_blocks = (num_sources + blockDim.x - 1) / blockDim.x;
    for (int sb = 0; sb < num_source_blocks; ++sb)
    {
        /* global source index */
        int t = sb * blockDim.x + threadIdx.x;

        /* Cache common values: StationI Jones matrices & source parameters */
        if (t < num_sources)
        {
            stationI[threadIdx.x] = jones[num_sources * blockIdx.x + t];
            sourceI[threadIdx.x]  = source_I[t];
            sourceQ[threadIdx.x]  = source_Q[t];
            sourceU[threadIdx.x]  = source_U[t];
            sourceV[threadIdx.x]  = source_V[t];
            sourceL[threadIdx.x]  = source_l[t];
            sourceM[threadIdx.x]  = source_m[t];
            sourceN[threadIdx.x]  = source_n[t];
        }

        /* Loop over other that make up the baseline handled by this thread block */
        for (int j = 0; j < STATION_BLOCK_SIZE; ++j)
        {
            /* global J station index */
            int idxStationJ = sJ0 + (blockIdx.y * STATION_BLOCK_SIZE) + j;
            if (idxStationJ < num_stations)
            {
                const float4c * __restrict__ stationJ = &jones[num_sources * idxStationJ];

                float l = sourceL[threadIdx.x];
                float m = sourceM[threadIdx.x];
                float n = sourceN[threadIdx.x];

                /* Compute bandwidth-smearing term. */
                float rb = oskar_sinc_f(uu[j] * l + vv[j] * m);

                /* Compute time-smearing term. */
                float rt = oskar_sinc_f(du_dt[j] * l + dv_dt[j] * m + dw_dt[j] * n);

                rb *= rt; /* smearing term */

                /* Accumulate baseline visibility response for source. */
                if (t < num_sources)
                {
                    float4c m1 = stationI[threadIdx.x];
                    float4c m2;
                    float I = sourceI[threadIdx.x];
                    float Q = sourceQ[threadIdx.x];
                    m2.a.x = I + Q;
                    m2.b.x = sourceU[threadIdx.x];
                    m2.b.y = sourceV[threadIdx.x];
                    m2.d.x = I - Q;

                    /* Multiply first Jones matrix with source brightness matrix. */
                    oskar_multiply_complex_matrix_hermitian_in_place_f(&m1, &m2);

                    /* Multiply result with second (Hermitian transposed) Jones matrix. */
                    m2 = stationJ[t];
                    oskar_multiply_complex_matrix_conjugate_transpose_in_place_f(&Vpq_source[threadIdx.x], &m2);

                    /* multiply by the smearing term */
                    Vpq_source[threadIdx.x].a.x *= rb;
                    Vpq_source[threadIdx.x].a.y *= rb;
                    Vpq_source[threadIdx.x].b.x *= rb;
                    Vpq_source[threadIdx.x].b.y *= rb;
                    Vpq_source[threadIdx.x].c.x *= rb;
                    Vpq_source[threadIdx.x].c.y *= rb;
                    Vpq_source[threadIdx.x].d.x *= rb;
                    Vpq_source[threadIdx.x].d.y *= rb;
                } /* Accumulate for the source */
            } /* (idxStationJ < num_stations) */

            /* Make sure all threads have finished for their source. */
            __syncthreads();

            /* Accumulate per source visibilities into per chunk visibilities */
            if (threadIdx.x == 0 && idxStationJ < num_stations)
            {
                for (int i = 0; i < blockDim.x; ++i)
                {
                    int tacc = (sb*num_sources) + i;
                    if (tacc < num_sources)
                    {
                        Vpq_sum[j].a.x += Vpq_source[i].a.x;
                        Vpq_sum[j].a.y += Vpq_source[i].a.y;
                        Vpq_sum[j].b.x += Vpq_source[i].b.x;
                        Vpq_sum[j].b.y += Vpq_source[i].b.y;
                        Vpq_sum[j].c.x += Vpq_source[i].c.x;
                        Vpq_sum[j].c.y += Vpq_source[i].c.y;
                        Vpq_sum[j].d.x += Vpq_source[i].d.x;
                        Vpq_sum[j].d.y += Vpq_source[i].d.y;
                    }
                }
            } /* Accumulate to baseline for the source chunk */
        } /* Loop over other stations that make up the baseline (I-J) */
    } /* Loop over blocks of sources */

    /* Write final visibilities to global memory */
    if (threadIdx.x < STATION_BLOCK_SIZE)
    {
        /* Index of baseline visibility */
        int sJ = sJ0 + (blockIdx.y * STATION_BLOCK_SIZE) + threadIdx.x;
        int idx = sJ * (num_stations-1) - (sJ-1) * sJ/2 + blockIdx.x - sJ -1;
        /* Add result of this thread block to the baseline visibility. */
        vis[idx] = Vpq_sum[threadIdx.x];
    }
} /* version 2 */


/* Double precision. */
__global__
void oskar_correlate_point_time_smearing_cudak_d(const int num_sources,
        const int num_stations, const double4c* jones, const double* source_I,
        const double* source_Q, const double* source_U, const double* source_V,
        const double* source_l, const double* source_m, const double* source_n,
        const double* station_u, const double* station_v,
        const double* station_x, const double* station_y, const double freq_hz,
        const double bandwidth_hz, const double time_int_sec,
        const double gha0_rad, const double dec0_rad, double4c* vis)
{
    /* Local variables. */
    double4c sum;
    double l, m, n, r1, r2;
    int i;

    /* Common values per thread block. */
    __device__ __shared__ double uu, vv, du_dt, dv_dt, dw_dt;
    __device__ __shared__ const double4c *station_i, *station_j;

    /* Return immediately if in the wrong half of the visibility matrix. */
    if (SJ >= SI) return;

    /* Use thread 0 to set up the block. */
    if (threadIdx.x == 0)
    {
        double temp;

        /* Baseline lengths. */
        uu = (station_u[SI] - station_u[SJ]) * 0.5;
        vv = (station_v[SI] - station_v[SJ]) * 0.5;

        /* Modify the baseline distance to include the common components
         * of the bandwidth smearing term. */
        temp = bandwidth_hz / freq_hz; /* Fractional bandwidth */
        uu *= temp;
        vv *= temp;

        /* Compute the derivatives for time-average smearing. */
        {
            double xx, yy, rot_angle;
            double sin_HA, cos_HA, sin_Dec, cos_Dec;
            sincos(gha0_rad, &sin_HA, &cos_HA);
            sincos(dec0_rad, &sin_Dec, &cos_Dec);
            xx = (station_x[SI] - station_x[SJ]) * 0.5;
            yy = (station_y[SI] - station_y[SJ]) * 0.5;
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
    smem_d4c[threadIdx.x] = sum;
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
            sum.a.x += smem_d4c[i].a.x;
            sum.a.y += smem_d4c[i].a.y;
            sum.b.x += smem_d4c[i].b.x;
            sum.b.y += smem_d4c[i].b.y;
            sum.c.x += smem_d4c[i].c.x;
            sum.c.y += smem_d4c[i].c.y;
            sum.d.x += smem_d4c[i].d.x;
            sum.d.y += smem_d4c[i].d.y;
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
