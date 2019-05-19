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
#include "correlate/oskar_cross_correlate_scalar_cuda.h"
#include "math/define_multiply.h"
#include "utility/oskar_kernel_macros.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

enum {
    VER_UNKNOWN    = -1,  // Not checked.
    VER_NONE       =  0,  // Not specified.
    // Actual versions:
    VER_OLD        =  1,
    VER_NON_SM     =  2,
    VER_SM         =  3
};
static int ver_specified_ = VER_UNKNOWN;
static int ver_cc_ = VER_UNKNOWN;
static int correlate_version(bool prec_double,
        double frac_bandwidth, double time_int_sec);

// Original kernel.
template
<
// Compile-time parameters.
bool BANDWIDTH_SMEARING, bool TIME_SMEARING, bool GAUSSIAN,
typename FP, typename FP2
>
OSKAR_XCORR_SCALAR_GPU(oskar_xcorr_scalar_cudak, BANDWIDTH_SMEARING, TIME_SMEARING, GAUSSIAN, FP, FP2)

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
typename FP, typename FP2
>
__global__
void oskar_xcorr_scalar_NON_SM_cudak(
        OSKAR_XCORR_ARGS(FP)
        const FP2* const __restrict__ jones,
        FP2*             __restrict__ vis)
{
    (void) src_Q;
    (void) src_U;
    (void) src_V;
    __shared__ FP uv_len[OKN_BPK], uu[OKN_BPK], vv[OKN_BPK], ww[OKN_BPK];
    __shared__ FP uu2[OKN_BPK], vv2[OKN_BPK], uuvv[OKN_BPK];
    __shared__ FP du[OKN_BPK], dv[OKN_BPK], dw[OKN_BPK];
    __shared__ const FP2 *st_q[OKN_BPK];
    FP2 t1, t2, sum;

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
    const FP2* const __restrict__ st_p = &jones[num_src * SP];

    // Each thread from given warp loops over a subset of the sources,
    // and each warp works with a different station q.
    sum.x = sum.y = (FP)0;
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

        // Multiply Jones scalars.
        smearing *= src_I[s];
        t1 = st_p[s];
        t2 = (st_q[w])[s];
        OSKAR_MUL_COMPLEX_CONJUGATE_IN_PLACE(FP2, t1, t2)

        // Multiply result by smearing term and accumulate.
        sum.x += t1.x * smearing; sum.y += t1.y * smearing;
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

            // Multiply Jones scalars.
            smearing *= src_I[s];
            t1 = st_p[s];
            t2 = (st_q[w])[s];
            OSKAR_MUL_COMPLEX_CONJUGATE_IN_PLACE(FP2, t1, t2)

            // Multiply result by smearing term and accumulate.
            sum.x += t1.x * smearing; sum.y += t1.y * smearing;
        }
    }

    // Reduce results within warp.
    WARP_REDUCE(sum.x);
    WARP_REDUCE(sum.y);

    // Add result of this warp to the baseline visibility.
    if (i == 0 && (OKN_BPK * SQ + w) < SP)
    {
        if (uv_len[w] < uv_min_lambda || uv_len[w] > uv_max_lambda) return;
        const int q = OKN_BPK * SQ + w;
        const int j = OSKAR_BASELINE_INDEX(num_stations, SP, q) + offset_out;
        vis[j].x += sum.x; vis[j].y += sum.y;
    }
}

template
<
// Compile-time parameters.
bool BANDWIDTH_SMEARING, bool TIME_SMEARING, bool GAUSSIAN,
typename FP, typename FP2
>
__global__
void oskar_xcorr_scalar_SM_cudak(
        OSKAR_XCORR_ARGS(FP)
        const FP2* const __restrict__ jones,
        FP2*             __restrict__ vis)
{
    (void) src_Q;
    (void) src_U;
    (void) src_V;
    __shared__ FP uv_len[OKN_BPK], uu[OKN_BPK], vv[OKN_BPK], ww[OKN_BPK];
    __shared__ FP uu2[OKN_BPK], vv2[OKN_BPK], uuvv[OKN_BPK];
    __shared__ FP du[OKN_BPK], dv[OKN_BPK], dw[OKN_BPK];
    __shared__ const FP2 *st_q[OKN_BPK];
    __shared__ FP   s_I[OKN_NSOURCES];
    __shared__ FP   s_l[OKN_NSOURCES];
    __shared__ FP   s_m[OKN_NSOURCES];
    __shared__ FP   s_n[OKN_NSOURCES];
    __shared__ FP   s_a[OKN_NSOURCES];
    __shared__ FP   s_b[OKN_NSOURCES];
    __shared__ FP   s_c[OKN_NSOURCES];
    __shared__ FP2 s_sp[OKN_NSOURCES];
    FP2 t1, t2, sum;

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
    const FP2* const __restrict__ st_p = &jones[num_src * SP];

    // Each thread from given warp loops over a subset of the sources,
    // and each warp works with a different station q.
    sum.x = sum.y = (FP)0;
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
            if (BANDWIDTH_SMEARING || TIME_SMEARING)
                s_m[i] = src_m[s];
            if (GAUSSIAN)
                s_c[i] = src_c[s];
        }
        if (w == 2)
        {
            if (BANDWIDTH_SMEARING || TIME_SMEARING)
                s_n[i] = src_n[s];
        }
        if (w == 3)
        {
            s_sp[i] = st_p[s];
        }
        __syncthreads();

        FP smearing;
        if (GAUSSIAN)
        {
            const FP t = s_a[i] * uu2[w] + s_b[i] * uuvv[w] + s_c[i] * vv2[w];
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

        // Multiply Jones scalars.
        smearing *= s_I[i];
        t1 = s_sp[i];
        t2 = (st_q[w])[s];
        OSKAR_MUL_COMPLEX_CONJUGATE_IN_PLACE(FP2, t1, t2)

        // Multiply result by smearing term and accumulate.
        sum.x += t1.x * smearing; sum.y += t1.y * smearing;
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
                if (BANDWIDTH_SMEARING || TIME_SMEARING)
                    s_m[i] = src_m[s];
                if (GAUSSIAN)
                    s_c[i] = src_c[s];
            }
            if (w == 2)
            {
                if (BANDWIDTH_SMEARING || TIME_SMEARING)
                    s_n[i] = src_n[s];
            }
            if (w == 3)
            {
                s_sp[i] = st_p[s];
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

            // Multiply Jones scalars.
            smearing *= s_I[i];
            t1 = s_sp[i];
            t2 = (st_q[w])[s];
            OSKAR_MUL_COMPLEX_CONJUGATE_IN_PLACE(FP2, t1, t2)

            // Multiply result by smearing term and accumulate.
            sum.x += t1.x * smearing; sum.y += t1.y * smearing;
        }
    }

    // Reduce results within warp.
    WARP_REDUCE(sum.x);
    WARP_REDUCE(sum.y);

    // Add result of this warp to the baseline visibility.
    if (i == 0 && (OKN_BPK * SQ + w) < SP)
    {
        if (uv_len[w] < uv_min_lambda || uv_len[w] > uv_max_lambda) return;
        const int q = OKN_BPK * SQ + w;
        const int j = OSKAR_BASELINE_INDEX(num_stations, SP, q) + offset_out;
        vis[j].x += sum.x; vis[j].y += sum.y;
    }
}

#define XCORR_KERNEL(NAME, BS, TS, GAUSSIAN, FP, FP2)\
        NAME<BS, TS, GAUSSIAN, FP, FP2>\
        OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)\
        (num_sources, num_stations, offset_out, d_I, 0, 0, 0, d_l, d_m, d_n,\
                d_a, d_b, d_c, d_station_u, d_station_v, d_station_w,\
                d_station_x, d_station_y, uv_min_lambda, uv_max_lambda,\
                inv_wavelength, frac_bandwidth, time_int_sec,\
                gha0_rad, dec0_rad, d_jones, d_vis);

#define XCORR_SELECT(NAME, GAUSSIAN, FP, FP2)\
        if (frac_bandwidth == (FP)0 && time_int_sec == (FP)0)\
            XCORR_KERNEL(NAME, false, false, GAUSSIAN, FP, FP2)\
        else if (frac_bandwidth != (FP)0 && time_int_sec == (FP)0)\
            XCORR_KERNEL(NAME, true, false, GAUSSIAN, FP, FP2)\
        else if (frac_bandwidth == (FP)0 && time_int_sec != (FP)0)\
            XCORR_KERNEL(NAME, false, true, GAUSSIAN, FP, FP2)\
        else if (frac_bandwidth != (FP)0 && time_int_sec != (FP)0)\
            XCORR_KERNEL(NAME, true, true, GAUSSIAN, FP, FP2)

void oskar_cross_correlate_scalar_point_cuda_f(
        int num_sources, int num_stations, int offset_out,
        const float2* d_jones, const float* d_I, const float* d_l,
        const float* d_m, const float* d_n,
        const float* d_station_u, const float* d_station_v,
        const float* d_station_w, const float* d_station_x,
        const float* d_station_y, float uv_min_lambda, float uv_max_lambda,
        float inv_wavelength, float frac_bandwidth, const float time_int_sec,
        const float gha0_rad, const float dec0_rad, float2* d_vis)
{
    const dim3 num_threads(128, 1);
    const float *d_a = 0, *d_b = 0, *d_c = 0;
    const int ver = correlate_version(false, frac_bandwidth, time_int_sec);
    if (ver == VER_NON_SM)
    {
        dim3 num_blocks(num_stations, (num_stations + OKN_BPK - 1) / OKN_BPK);
        const size_t shared_mem = 0;
        XCORR_SELECT(oskar_xcorr_scalar_NON_SM_cudak, false, float, float2)
    }
    else if (ver == VER_SM)
    {
        dim3 num_blocks(num_stations, (num_stations + OKN_BPK - 1) / OKN_BPK);
        const size_t shared_mem = 0;
        XCORR_SELECT(oskar_xcorr_scalar_SM_cudak, false, float, float2)
    }
    else
    {
        dim3 num_blocks(num_stations, num_stations);
        const size_t shared_mem = num_threads.x * sizeof(float2);
        XCORR_SELECT(oskar_xcorr_scalar_cudak, false, float, float2)
    }
}

void oskar_cross_correlate_scalar_point_cuda_d(
        int num_sources, int num_stations, int offset_out,
        const double2* d_jones, const double* d_I, const double* d_l,
        const double* d_m, const double* d_n,
        const double* d_station_u, const double* d_station_v,
        const double* d_station_w, const double* d_station_x,
        const double* d_station_y, double uv_min_lambda, double uv_max_lambda,
        double inv_wavelength, double frac_bandwidth, const double time_int_sec,
        const double gha0_rad, const double dec0_rad, double2* d_vis)
{
    const dim3 num_threads(128, 1);
    const double *d_a = 0, *d_b = 0, *d_c = 0;
    const int ver = correlate_version(true, frac_bandwidth, time_int_sec);
    if (ver == VER_NON_SM)
    {
        dim3 num_blocks(num_stations, (num_stations + OKN_BPK - 1) / OKN_BPK);
        const size_t shared_mem = 0;
        XCORR_SELECT(oskar_xcorr_scalar_NON_SM_cudak, false, double, double2)
    }
    else if (ver == VER_SM)
    {
        dim3 num_blocks(num_stations, (num_stations + OKN_BPK - 1) / OKN_BPK);
        const size_t shared_mem = 0;
        XCORR_SELECT(oskar_xcorr_scalar_SM_cudak, false, double, double2)
    }
    else
    {
        dim3 num_blocks(num_stations, num_stations);
        const size_t shared_mem = num_threads.x * sizeof(double2);
        XCORR_SELECT(oskar_xcorr_scalar_cudak, false, double, double2)
    }
}

void oskar_cross_correlate_scalar_gaussian_cuda_f(
        int num_sources, int num_stations, int offset_out,
        const float2* d_jones, const float* d_I, const float* d_l,
        const float* d_m, const float* d_n,
        const float* d_a, const float* d_b,
        const float* d_c, const float* d_station_u,
        const float* d_station_v, const float* d_station_w,
        const float* d_station_x, const float* d_station_y,
        float uv_min_lambda, float uv_max_lambda, float inv_wavelength,
        float frac_bandwidth, float time_int_sec, float gha0_rad,
        float dec0_rad, float2* d_vis)
{
    const dim3 num_threads(128, 1);
    const int ver = correlate_version(false, frac_bandwidth, time_int_sec);
    if (ver == VER_NON_SM)
    {
        dim3 num_blocks(num_stations, (num_stations + OKN_BPK - 1) / OKN_BPK);
        const size_t shared_mem = 0;
        XCORR_SELECT(oskar_xcorr_scalar_NON_SM_cudak, true, float, float2)
    }
    else if (ver == VER_SM)
    {
        dim3 num_blocks(num_stations, (num_stations + OKN_BPK - 1) / OKN_BPK);
        const size_t shared_mem = 0;
        XCORR_SELECT(oskar_xcorr_scalar_SM_cudak, true, float, float2)
    }
    else
    {
        dim3 num_blocks(num_stations, num_stations);
        const size_t shared_mem = num_threads.x * sizeof(float2);
        XCORR_SELECT(oskar_xcorr_scalar_cudak, true, float, float2)
    }
}

void oskar_cross_correlate_scalar_gaussian_cuda_d(
        int num_sources, int num_stations, int offset_out,
        const double2* d_jones, const double* d_I, const double* d_l,
        const double* d_m, const double* d_n,
        const double* d_a, const double* d_b,
        const double* d_c, const double* d_station_u,
        const double* d_station_v, const double* d_station_w,
        const double* d_station_x, const double* d_station_y,
        double uv_min_lambda, double uv_max_lambda, double inv_wavelength,
        double frac_bandwidth, double time_int_sec, double gha0_rad,
        double dec0_rad, double2* d_vis)
{
    const dim3 num_threads(128, 1);
    const int ver = correlate_version(true, frac_bandwidth, time_int_sec);
    if (ver == VER_NON_SM)
    {
        dim3 num_blocks(num_stations, (num_stations + OKN_BPK - 1) / OKN_BPK);
        const size_t shared_mem = 0;
        XCORR_SELECT(oskar_xcorr_scalar_NON_SM_cudak, true, double, double2)
    }
    else if (ver == VER_SM)
    {
        dim3 num_blocks(num_stations, (num_stations + OKN_BPK - 1) / OKN_BPK);
        const size_t shared_mem = 0;
        XCORR_SELECT(oskar_xcorr_scalar_SM_cudak, true, double, double2)
    }
    else
    {
        dim3 num_blocks(num_stations, num_stations);
        const size_t shared_mem = num_threads.x * sizeof(double2);
        XCORR_SELECT(oskar_xcorr_scalar_cudak, true, double, double2)
    }
}

int correlate_version(bool prec_double,
        double frac_bandwidth, double time_int_sec)
{
    // Check the environment variable if necessary
    // and use the specified version if it has been set.
    if (ver_specified_ == VER_UNKNOWN)
    {
        const char* v = getenv("OSKAR_CORRELATE");
        if (v)
        {
            if (!strcmp(v, "OLD") || !strcmp(v, "old"))
                ver_specified_ = VER_OLD;
            else if (!strcmp(v, "SM") || !strcmp(v, "sm"))
                ver_specified_ = VER_SM;
            else if (strstr(v, "NO") || strstr(v, "no"))
                ver_specified_ = VER_NON_SM;
        }
        if (ver_specified_ == VER_UNKNOWN)
            ver_specified_ = VER_NONE;
    }
    if (ver_specified_ > VER_NONE) return ver_specified_;

    // Check the device compute capability if required.
    if (ver_cc_ == VER_UNKNOWN)
    {
        int ma = 0, mi = 0, id = 0;
        cudaGetDevice(&id);
        cudaDeviceGetAttribute(&ma, cudaDevAttrComputeCapabilityMajor, id);
        cudaDeviceGetAttribute(&mi, cudaDevAttrComputeCapabilityMinor, id);
        ver_cc_ = 10 * ma + mi;
    }

    // Use non-shared-memory version on Volta.
    if (ver_cc_ >= 70) return VER_NON_SM;

    // Decide which is the best version to use on pre-Volta architectures.
    if (ver_cc_ >= 30)
    {
        const bool smearing = (frac_bandwidth != 0.0 || time_int_sec != 0.0);
        if (prec_double && smearing) return VER_NON_SM;
        if (prec_double && !smearing) return VER_SM;
        if (!prec_double && !smearing) return VER_NON_SM;
    }

    // Otherwise, use the old version.
    return VER_OLD;
}
