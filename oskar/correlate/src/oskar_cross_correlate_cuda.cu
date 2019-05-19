/*
 * Copyright (c) 2011-2019, The University of Oxford
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

#include "correlate/define_correlate_utils.h"
#include "correlate/define_cross_correlate.h"
#include "correlate/oskar_cross_correlate_cuda.h"
#include "math/define_multiply.h"
#include "utility/oskar_kernel_macros.h"
#include "utility/oskar_vector_types.h"

#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

enum { VER_OLD = 1, VER_NON_SM = 2, VER_SM = 3 };
static int ver_ = 0;
static int correlate_version(void);

// Original kernel.
template
<
// Compile-time parameters.
bool BANDWIDTH_SMEARING, bool TIME_SMEARING, bool GAUSSIAN,
typename FP, typename FP2, typename FP4c
>
OSKAR_XCORR_GPU(oskar_xcorr_cudak, BANDWIDTH_SMEARING, TIME_SMEARING, GAUSSIAN, FP, FP2, FP4c)

// Indices into the visibility/baseline matrix.
#define SP blockIdx.x /* Column index. */
#define SQ blockIdx.y /* Row index. */

#define OKN_NSOURCES 32
#define OKN_BPK 4 /* baselines per kernel */
#define WARP 32

template
<
// Compile-time parameters.
bool BANDWIDTH_SMEARING, bool TIME_SMEARING, bool GAUSSIAN,
typename FP, typename FP2, typename FP4c
>
__global__
void oskar_xcorr_NON_SM_cudak(
        OSKAR_XCORR_ARGS(FP)
        const FP4c* const __restrict__ jones,
        FP4c*             __restrict__ vis)
{
    __shared__ FP uv_len[OKN_BPK], uu[OKN_BPK], vv[OKN_BPK], ww[OKN_BPK];
    __shared__ FP uu2[OKN_BPK], vv2[OKN_BPK], uuvv[OKN_BPK];
    __shared__ FP du[OKN_BPK], dv[OKN_BPK], dw[OKN_BPK];
    __shared__ const FP4c *st_q[OKN_BPK];
    FP4c m1, m2, sum;

    const int w = (threadIdx.x >> 5); // Warp ID.
    const int i = (threadIdx.x & 31); // ID within warp (local ID).

    // Return immediately if in the wrong half of the visibility matrix.
    if (OKN_BPK * SQ >= SP) return;

    // Get baseline values per warp.
    if (i == 0)
    {
        const int i_sq = OKN_BPK * SQ + w;

        // Set pointer to source vector for station q to safe position
        // so non-existence SQ >= SP does not cause problems.
        st_q[w] = &jones[0];

        if (i_sq < num_stations)
        {
            OSKAR_BASELINE_TERMS(FP, st_u[SP], st_u[i_sq], st_v[SP], st_v[i_sq],
                    st_w[SP], st_w[i_sq], uu[w], vv[w], ww[w],
                    uu2[w], vv2[w], uuvv[w], uv_len[w]);

            if (TIME_SMEARING)
                OSKAR_BASELINE_DELTAS(FP, st_x[SP], st_x[i_sq],
                        st_y[SP], st_y[i_sq], du[w], dv[w], dw[w]);

            // Get valid pointer to source vector for station q.
            st_q[w] = &jones[num_src * i_sq];
        }
    }
    __syncthreads();

    // Get pointer to source vector for station p.
    const FP4c* const __restrict__ st_p = &jones[num_src * SP];

    // Each thread from given warp loops over a subset of the sources,
    // and each warp works with a different station q.
    OSKAR_CLEAR_COMPLEX_MATRIX(FP, sum)
    const int itemp = (num_src >> 5) * WARP;
    for (int s = i; s < itemp; s += WARP)
    {
        FP smearing;
        if (GAUSSIAN)
        {
            const FP t = src_a[s] * uu2[w] +
                    src_b[s] * uuvv[w] + src_c[s] * vv2[w];
            smearing = exp((FP) -t);
        }
        else smearing = (FP) 1;
        if (BANDWIDTH_SMEARING || TIME_SMEARING)
        {
            const FP l = src_l[s], m = src_m[s], n = src_n[s] - (FP) 1;
            if (BANDWIDTH_SMEARING)
            {
                const FP t = uu[w] * l + vv[w] * m + ww[w] * n;
                smearing *= OSKAR_SINC(FP, t);
            }
            if (TIME_SMEARING)
            {
                const FP t = du[w] * l + dv[w] * m + dw[w] * n;
                smearing *= OSKAR_SINC(FP, t);
            }
        }

        // Construct source brightness matrix.
        OSKAR_CONSTRUCT_B(FP, m2, src_I[s], src_Q[s],
                src_U[s], src_V[s])

        // Multiply first Jones matrix with source brightness matrix.
        m1 = st_p[s]; // FIXME(FD) Use OSKAR_LOAD_MATRIX here?
        OSKAR_MUL_COMPLEX_MATRIX_HERMITIAN_IN_PLACE(FP2, m1, m2)

        // Multiply result with second (Hermitian transposed) Jones matrix.
        OSKAR_LOAD_MATRIX(m2, (st_q[w])[s])
        OSKAR_MUL_COMPLEX_MATRIX_CONJUGATE_TRANSPOSE_IN_PLACE(FP2, m1, m2)

        // Multiply result by smearing term and accumulate.
        OSKAR_MUL_ADD_COMPLEX_MATRIX_SCALAR(sum, m1, smearing)
        __syncthreads();
    }
    if ((num_src & 31) > 0)
    {
        int s = (num_src >> 5) * WARP + i;
        if (s < num_src)
        {
            FP smearing;
            if (GAUSSIAN)
            {
                const FP t = src_a[s] * uu2[w] +
                        src_b[s] * uuvv[w] + src_c[s] * vv2[w];
                smearing = exp((FP) -t);
            }
            else smearing = (FP) 1;
            if (BANDWIDTH_SMEARING || TIME_SMEARING)
            {
                const FP l = src_l[s], m = src_m[s], n = src_n[s] - (FP) 1;
                if (BANDWIDTH_SMEARING)
                {
                    const FP t = uu[w] * l + vv[w] * m + ww[w] * n;
                    smearing *= OSKAR_SINC(FP, t);
                }
                if (TIME_SMEARING)
                {
                    const FP t = du[w] * l + dv[w] * m + dw[w] * n;
                    smearing *= OSKAR_SINC(FP, t);
                }
            }

            // Construct source brightness matrix.
            OSKAR_CONSTRUCT_B(FP, m2, src_I[s], src_Q[s],
                    src_U[s], src_V[s])

            // Multiply first Jones matrix with source brightness matrix.
            m1 = st_p[s]; // FIXME(FD) Use OSKAR_LOAD_MATRIX here?
            OSKAR_MUL_COMPLEX_MATRIX_HERMITIAN_IN_PLACE(FP2, m1, m2)

            // Multiply result with second (Hermitian transposed) Jones matrix.
            OSKAR_LOAD_MATRIX(m2, (st_q[w])[s])
            OSKAR_MUL_COMPLEX_MATRIX_CONJUGATE_TRANSPOSE_IN_PLACE(FP2, m1, m2)

            // Multiply result by smearing term and accumulate.
            OSKAR_MUL_ADD_COMPLEX_MATRIX_SCALAR(sum, m1, smearing)
        }
    }

    // Reduce matrices within warp.
    WARP_REDUCE(sum.a.x);
    WARP_REDUCE(sum.a.y);
    WARP_REDUCE(sum.b.x);
    WARP_REDUCE(sum.b.y);
    WARP_REDUCE(sum.c.x);
    WARP_REDUCE(sum.c.y);
    WARP_REDUCE(sum.d.x);
    WARP_REDUCE(sum.d.y);

    // Add result of this warp to the baseline visibility.
    if (i == 0 && (OKN_BPK * SQ + w) < SP)
    {
        if (uv_len[w] < uv_min_lambda || uv_len[w] > uv_max_lambda) return;
        const int q = OKN_BPK * SQ + w;
        const int j = OSKAR_BASELINE_INDEX(num_stations, SP, q) + offset_out;
        OSKAR_ADD_COMPLEX_MATRIX_IN_PLACE(vis[j], sum);
    }
}

template
<
// Compile-time parameters.
bool BANDWIDTH_SMEARING, bool TIME_SMEARING, bool GAUSSIAN,
typename FP, typename FP2, typename FP4c
>
__global__
void oskar_xcorr_SM_cudak(
        OSKAR_XCORR_ARGS(FP)
        const FP4c* const __restrict__ jones,
        FP4c*             __restrict__ vis)
{
    __shared__ FP uv_len[OKN_BPK], uu[OKN_BPK], vv[OKN_BPK], ww[OKN_BPK];
    __shared__ FP uu2[OKN_BPK], vv2[OKN_BPK], uuvv[OKN_BPK];
    __shared__ FP du[OKN_BPK], dv[OKN_BPK], dw[OKN_BPK];
    __shared__ const FP4c *st_q[OKN_BPK];
    __shared__ FP   s_I[OKN_NSOURCES];
    __shared__ FP   s_Q[OKN_NSOURCES];
    __shared__ FP   s_U[OKN_NSOURCES];
    __shared__ FP   s_V[OKN_NSOURCES];
    __shared__ FP   s_l[OKN_NSOURCES];
    __shared__ FP   s_m[OKN_NSOURCES];
    __shared__ FP   s_n[OKN_NSOURCES];
    __shared__ FP   s_a[OKN_NSOURCES];
    __shared__ FP   s_b[OKN_NSOURCES];
    __shared__ FP   s_c[OKN_NSOURCES];
    __shared__ FP4c s_sp[OKN_NSOURCES];
    FP4c m1, m2, sum;

    const int w = (threadIdx.x >> 5); // Warp ID.
    const int i = (threadIdx.x & 31); // ID within warp (local ID).

    // Return immediately if in the wrong half of the visibility matrix.
    if (OKN_BPK * SQ >= SP) return;

    // Get baseline values per warp.
    if (i == 0)
    {
        const int i_sq = OKN_BPK * SQ + w;

        // Set pointer to source vector for station q to safe position
        // so non-existence SQ >= SP does not cause problems.
        st_q[w] = &jones[0];

        if (i_sq < num_stations)
        {
            OSKAR_BASELINE_TERMS(FP, st_u[SP], st_u[i_sq], st_v[SP], st_v[i_sq],
                    st_w[SP], st_w[i_sq], uu[w], vv[w], ww[w],
                    uu2[w], vv2[w], uuvv[w], uv_len[w]);

            if (TIME_SMEARING)
                OSKAR_BASELINE_DELTAS(FP, st_x[SP], st_x[i_sq],
                        st_y[SP], st_y[i_sq], du[w], dv[w], dw[w]);

            // Get valid pointer to source vector for station q.
            st_q[w] = &jones[num_src * i_sq];
        }
    }
    __syncthreads();

    // Get pointer to source vector for station p.
    const FP4c* const __restrict__ st_p = &jones[num_src * SP];

    // Each thread from given warp loops over a subset of the sources,
    // and each warp works with a different station q.
    OSKAR_CLEAR_COMPLEX_MATRIX(FP, sum)
    const int itemp = (num_src >> 5) * WARP;
    for (int s = i; s < itemp; s += WARP)
    {
        if (w == 0)
        {
            s_I[i] = src_I[s];
            if (BANDWIDTH_SMEARING || TIME_SMEARING)
                s_l[i] = src_l[s];
            if (GAUSSIAN)
            {
                s_a[i] = src_a[s];
                s_b[i] = src_b[s];
            }
        }
        if (w == 1)
        {
            s_Q[i] = src_Q[s];
            if (BANDWIDTH_SMEARING || TIME_SMEARING)
                s_m[i] = src_m[s];
            if (GAUSSIAN)
                s_c[i] = src_c[s];
        }
        if (w == 2)
        {
            s_U[i] = src_U[s];
            s_V[i] = src_V[s];
            if (BANDWIDTH_SMEARING || TIME_SMEARING)
                s_n[i] = src_n[s];
        }
        if (w == 3)
        {
            s_sp[i] = st_p[s]; // FIXME(FD) Use OSKAR_LOAD_MATRIX here?
        }
        __syncthreads();

        FP smearing;
        if (GAUSSIAN)
        {
            const FP t = s_a[i] * uu2[w] +
                    s_b[i] * uuvv[w] + s_c[i] * vv2[w];
            smearing = exp((FP) -t);
        }
        else smearing = (FP) 1;
        if (BANDWIDTH_SMEARING || TIME_SMEARING)
        {
            const FP l = s_l[i], m = s_m[i], n = s_n[i] - (FP) 1;
            if (BANDWIDTH_SMEARING)
            {
                const FP t = uu[w] * l + vv[w] * m + ww[w] * n;
                smearing *= OSKAR_SINC(FP, t);
            }
            if (TIME_SMEARING)
            {
                const FP t = du[w] * l + dv[w] * m + dw[w] * n;
                smearing *= OSKAR_SINC(FP, t);
            }
        }

        // Construct source brightness matrix.
        OSKAR_CONSTRUCT_B(FP, m2, s_I[i], s_Q[i], s_U[i], s_V[i])

        // Multiply first Jones matrix with source brightness matrix.
        m1 = s_sp[i];
        OSKAR_MUL_COMPLEX_MATRIX_HERMITIAN_IN_PLACE(FP2, m1, m2)

        // Multiply result with second (Hermitian transposed) Jones matrix.
        OSKAR_LOAD_MATRIX(m2, (st_q[w])[s])
        OSKAR_MUL_COMPLEX_MATRIX_CONJUGATE_TRANSPOSE_IN_PLACE(FP2, m1, m2)

        // Multiply result by smearing term and accumulate.
        OSKAR_MUL_ADD_COMPLEX_MATRIX_SCALAR(sum, m1, smearing)
        __syncthreads();
    }
    if ((num_src & 31) > 0)
    {
        int s = (num_src >> 5) * WARP + i;
        if (s < num_src)
        {
            if (w == 0)
            {
                s_I[i] = src_I[s];
                if (BANDWIDTH_SMEARING || TIME_SMEARING)
                    s_l[i] = src_l[s];
                if (GAUSSIAN)
                {
                    s_a[i] = src_a[s];
                    s_b[i] = src_b[s];
                }
            }
            if (w == 1)
            {
                s_Q[i] = src_Q[s];
                if (BANDWIDTH_SMEARING || TIME_SMEARING)
                    s_m[i] = src_m[s];
                if (GAUSSIAN)
                    s_c[i] = src_c[s];
            }
            if (w == 2)
            {
                s_U[i] = src_U[s];
                s_V[i] = src_V[s];
                if (BANDWIDTH_SMEARING || TIME_SMEARING)
                    s_n[i] = src_n[s];
            }
            if (w == 3)
            {
                s_sp[i] = st_p[s]; // FIXME(FD) Use OSKAR_LOAD_MATRIX here?
            }
        }
        __syncthreads();
        if (s < num_src)
        {
            FP smearing;
            if (GAUSSIAN)
            {
                const FP t = s_a[i] * uu2[w] +
                        s_b[i] * uuvv[w] + s_c[i] * vv2[w];
                smearing = exp((FP) -t);
            }
            else smearing = (FP) 1;
            if (BANDWIDTH_SMEARING || TIME_SMEARING)
            {
                const FP l = s_l[i], m = s_m[i], n = s_n[i] - (FP) 1;
                if (BANDWIDTH_SMEARING)
                {
                    const FP t = uu[w] * l + vv[w] * m + ww[w] * n;
                    smearing *= OSKAR_SINC(FP, t);
                }
                if (TIME_SMEARING)
                {
                    const FP t = du[w] * l + dv[w] * m + dw[w] * n;
                    smearing *= OSKAR_SINC(FP, t);
                }
            }

            // Construct source brightness matrix.
            OSKAR_CONSTRUCT_B(FP, m2, s_I[i], s_Q[i], s_U[i], s_V[i])

            // Multiply first Jones matrix with source brightness matrix.
            m1 = s_sp[i];
            OSKAR_MUL_COMPLEX_MATRIX_HERMITIAN_IN_PLACE(FP2, m1, m2)

            // Multiply result with second (Hermitian transposed) Jones matrix.
            OSKAR_LOAD_MATRIX(m2, (st_q[w])[s])
            OSKAR_MUL_COMPLEX_MATRIX_CONJUGATE_TRANSPOSE_IN_PLACE(FP2, m1, m2)

            // Multiply result by smearing term and accumulate.
            OSKAR_MUL_ADD_COMPLEX_MATRIX_SCALAR(sum, m1, smearing)
        }
    }

    // Reduce matrices within warp.
    WARP_REDUCE(sum.a.x);
    WARP_REDUCE(sum.a.y);
    WARP_REDUCE(sum.b.x);
    WARP_REDUCE(sum.b.y);
    WARP_REDUCE(sum.c.x);
    WARP_REDUCE(sum.c.y);
    WARP_REDUCE(sum.d.x);
    WARP_REDUCE(sum.d.y);

    // Add result of this warp to the baseline visibility.
    if (i == 0 && (OKN_BPK * SQ + w) < SP)
    {
        if (uv_len[w] < uv_min_lambda || uv_len[w] > uv_max_lambda) return;
        const int q = OKN_BPK * SQ + w;
        const int j = OSKAR_BASELINE_INDEX(num_stations, SP, q) + offset_out;
        OSKAR_ADD_COMPLEX_MATRIX_IN_PLACE(vis[j], sum);
    }
}

#define XCORR_KERNEL(NAME, BS, TS, GAUSSIAN, FP, FP2, FP4c)\
        NAME<BS, TS, GAUSSIAN, FP, FP2, FP4c>\
        OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)\
        (num_sources, num_stations, offset_out, d_I, d_Q, d_U, d_V,\
                d_l, d_m, d_n, d_a, d_b, d_c,\
                d_station_u, d_station_v, d_station_w,\
                d_station_x, d_station_y, uv_min_lambda, uv_max_lambda,\
                inv_wavelength, frac_bandwidth, time_int_sec,\
                gha0_rad, dec0_rad, d_jones, d_vis);

#define XCORR_SELECT(NAME, GAUSSIAN, FP, FP2, FP4c)\
        if (frac_bandwidth == (FP)0 && time_int_sec == (FP)0)\
            XCORR_KERNEL(NAME, false, false, GAUSSIAN, FP, FP2, FP4c)\
        else if (frac_bandwidth != (FP)0 && time_int_sec == (FP)0)\
            XCORR_KERNEL(NAME, true, false, GAUSSIAN, FP, FP2, FP4c)\
        else if (frac_bandwidth == (FP)0 && time_int_sec != (FP)0)\
            XCORR_KERNEL(NAME, false, true, GAUSSIAN, FP, FP2, FP4c)\
        else if (frac_bandwidth != (FP)0 && time_int_sec != (FP)0)\
            XCORR_KERNEL(NAME, true, true, GAUSSIAN, FP, FP2, FP4c)

void oskar_cross_correlate_point_cuda_f(
        int num_sources, int num_stations, int offset_out,
        const float4c* d_jones, const float* d_I, const float* d_Q,
        const float* d_U, const float* d_V,
        const float* d_l, const float* d_m, const float* d_n,
        const float* d_station_u, const float* d_station_v,
        const float* d_station_w,
        const float* d_station_x, const float* d_station_y,
        float uv_min_lambda, float uv_max_lambda, float inv_wavelength,
        float frac_bandwidth, float time_int_sec, float gha0_rad,
        float dec0_rad, float4c* d_vis)
{
    const dim3 num_threads(128, 1);
    const float *d_a = 0, *d_b = 0, *d_c = 0;
    if (correlate_version() == VER_NON_SM)
    {
        dim3 num_blocks(num_stations, (num_stations + OKN_BPK - 1) / OKN_BPK);
        const size_t shared_mem = 0;
        XCORR_SELECT(oskar_xcorr_NON_SM_cudak, false, float, float2, float4c)
    }
    else if (correlate_version() == VER_SM)
    {
        dim3 num_blocks(num_stations, (num_stations + OKN_BPK - 1) / OKN_BPK);
        const size_t shared_mem = 0;
        XCORR_SELECT(oskar_xcorr_SM_cudak, false, float, float2, float4c)
    }
    else
    {
        dim3 num_blocks(num_stations, num_stations);
        const size_t shared_mem = num_threads.x * sizeof(float4c);
        XCORR_SELECT(oskar_xcorr_cudak, false, float, float2, float4c)
    }
}

void oskar_cross_correlate_point_cuda_d(
        int num_sources, int num_stations, int offset_out,
        const double4c* d_jones, const double* d_I, const double* d_Q,
        const double* d_U, const double* d_V,
        const double* d_l, const double* d_m, const double* d_n,
        const double* d_station_u, const double* d_station_v,
        const double* d_station_w,
        const double* d_station_x, const double* d_station_y,
        double uv_min_lambda, double uv_max_lambda, double inv_wavelength,
        double frac_bandwidth, double time_int_sec, double gha0_rad,
        double dec0_rad, double4c* d_vis)
{
    const double *d_a = 0, *d_b = 0, *d_c = 0;
    if (correlate_version() == VER_NON_SM)
    {
        dim3 num_threads(128, 1);
        dim3 num_blocks(num_stations, (num_stations + OKN_BPK - 1) / OKN_BPK);
        const size_t shared_mem = 0;
        XCORR_SELECT(oskar_xcorr_NON_SM_cudak, false, double, double2, double4c)
    }
    else if (correlate_version() == VER_SM)
    {
        dim3 num_threads(128, 1);
        dim3 num_blocks(num_stations, (num_stations + OKN_BPK - 1) / OKN_BPK);
        const size_t shared_mem = 0;
        XCORR_SELECT(oskar_xcorr_SM_cudak, false, double, double2, double4c)
    }
    else
    {
        dim3 num_threads(128, 1);
        dim3 num_blocks(num_stations, num_stations);
        if (time_int_sec != 0.0) num_threads.x = 64;
        const size_t shared_mem = num_threads.x * sizeof(double4c);
        XCORR_SELECT(oskar_xcorr_cudak, false, double, double2, double4c)
    }
}

void oskar_cross_correlate_gaussian_cuda_f(
        int num_sources, int num_stations, int offset_out,
        const float4c* d_jones, const float* d_I, const float* d_Q,
        const float* d_U, const float* d_V,
        const float* d_l, const float* d_m, const float* d_n,
        const float* d_a, const float* d_b, const float* d_c,
        const float* d_station_u, const float* d_station_v,
        const float* d_station_w, const float* d_station_x,
        const float* d_station_y, float uv_min_lambda, float uv_max_lambda,
        float inv_wavelength, float frac_bandwidth, float time_int_sec,
        float gha0_rad, float dec0_rad, float4c* d_vis)
{
    const dim3 num_threads(128, 1);
    if (correlate_version() == VER_NON_SM)
    {
        dim3 num_blocks(num_stations, (num_stations + OKN_BPK - 1) / OKN_BPK);
        const size_t shared_mem = 0;
        XCORR_SELECT(oskar_xcorr_NON_SM_cudak, true, float, float2, float4c)
    }
    else if (correlate_version() == VER_SM)
    {
        dim3 num_blocks(num_stations, (num_stations + OKN_BPK - 1) / OKN_BPK);
        const size_t shared_mem = 0;
        XCORR_SELECT(oskar_xcorr_SM_cudak, true, float, float2, float4c)
    }
    else
    {
        dim3 num_blocks(num_stations, num_stations);
        const size_t shared_mem = num_threads.x * sizeof(float4c);
        XCORR_SELECT(oskar_xcorr_cudak, true, float, float2, float4c)
    }
}

void oskar_cross_correlate_gaussian_cuda_d(
        int num_sources, int num_stations, int offset_out,
        const double4c* d_jones, const double* d_I, const double* d_Q,
        const double* d_U, const double* d_V,
        const double* d_l, const double* d_m, const double* d_n,
        const double* d_a, const double* d_b, const double* d_c,
        const double* d_station_u, const double* d_station_v,
        const double* d_station_w, const double* d_station_x,
        const double* d_station_y, double uv_min_lambda, double uv_max_lambda,
        double inv_wavelength, double frac_bandwidth, double time_int_sec,
        double gha0_rad, double dec0_rad, double4c* d_vis)
{
    if (correlate_version() == VER_NON_SM)
    {
        dim3 num_threads(128, 1);
        dim3 num_blocks(num_stations, (num_stations + OKN_BPK - 1) / OKN_BPK);
        const size_t shared_mem = 0;
        XCORR_SELECT(oskar_xcorr_NON_SM_cudak, true, double, double2, double4c)
    }
    else if (correlate_version() == VER_SM)
    {
        dim3 num_threads(128, 1);
        dim3 num_blocks(num_stations, (num_stations + OKN_BPK - 1) / OKN_BPK);
        const size_t shared_mem = 0;
        XCORR_SELECT(oskar_xcorr_SM_cudak, true, double, double2, double4c)
    }
    else
    {
        dim3 num_threads(128, 1);
        dim3 num_blocks(num_stations, num_stations);
        if (time_int_sec != 0.0) num_threads.x = 64;
        const size_t shared_mem = num_threads.x * sizeof(double4c);
        XCORR_SELECT(oskar_xcorr_cudak, true, double, double2, double4c)
    }
}

int correlate_version()
{
    if (ver_ == 0)
    {
        const char* v = getenv("OSKAR_CORRELATE");
        if (v)
        {
            if (!strcmp(v, "OLD") || !strcmp(v, "old"))
                ver_ = VER_OLD;
            else if (!strcmp(v, "SM") || !strcmp(v, "sm"))
                ver_ = VER_SM;
            else if (strstr(v, "NO") || strstr(v, "no"))
                ver_ = VER_NON_SM;
        }
        if (ver_ == 0)
        {
            int ma = 0, mi = 0, id = 0;
            cudaGetDevice(&id);
            cudaDeviceGetAttribute(&ma, cudaDevAttrComputeCapabilityMajor, id);
            cudaDeviceGetAttribute(&mi, cudaDevAttrComputeCapabilityMinor, id);
            const int compute = 10 * ma + mi;
            if (compute >= 70)
                ver_ = VER_NON_SM;
            else if (compute >= 30)
                ver_ = VER_SM;
            else
                ver_ = VER_OLD;
        }
    }
    return ver_;
}
