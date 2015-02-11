/*
 * Copyright (c) 2015, The University of Oxford
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

#include <oskar_correlate_functions_inline.h>
#include <oskar_correlate_point_time_smearing_new_cuda.h>
#include <oskar_add_inline.h>
#include <cstdio>

/* Kernels. ================================================================ */

#define BLOCK_SINGLE 8
#define BLOCK_DOUBLE 8

/* Single precision. */

#define ROW threadIdx.y
#define COL threadIdx.x

__global__
void oskar_correlate_point_time_smearing_new_cudak_f(const int num_sources,
        const int num_stations, const float4c* restrict jones,
        const float* restrict source_l, const float* restrict source_m,
        const float* restrict source_n, const float* restrict station_u,
        const float* restrict station_v, const float* restrict station_w,
        const float* restrict station_x, const float* restrict station_y,
        const float uv_min_lambda, const float uv_max_lambda,
        const float inv_wavelength, const float frac_bandwidth,
        const float time_int_sec, const float gha0_rad, const float dec0_rad,
        float4c* restrict vis)
{
    __shared__ float4c As[BLOCK_SINGLE][BLOCK_SINGLE];
    __shared__ float4c Bs[BLOCK_SINGLE][BLOCK_SINGLE];
    __shared__ float ls[BLOCK_SINGLE], ms[BLOCK_SINGLE], ns[BLOCK_SINGLE];
    float uv_len, uu, vv, ww, uu2, vv2, uuvv, du_dt, dv_dt, dw_dt;

    /* Exit if block is in the wrong half of the visibility matrix. */
    if (blockIdx.y > blockIdx.x) return;

    /* Each thread block computes one sub-matrix of the visibility matrix.
     * Each thread computes visibilities for one baseline. */
    float4c sum;
    oskar_clear_complex_matrix_f(&sum);

    /* Get station indices. */
    const int iSP = blockIdx.x * BLOCK_SINGLE + COL;
    const int iSQ = blockIdx.y * BLOCK_SINGLE + ROW;

    /* Evaluate per-baseline terms. */
    if (iSP < num_stations && iSQ < num_stations)
    {
        oskar_evaluate_baseline_terms_inline_f(station_u[iSP],
                station_u[iSQ], station_v[iSP], station_v[iSQ],
                station_w[iSP], station_w[iSQ], inv_wavelength,
                frac_bandwidth, &uv_len, &uu, &vv, &ww, &uu2, &vv2, &uuvv);
        oskar_evaluate_baseline_derivatives_inline_f(station_x[iSP],
                station_x[iSQ], station_y[iSP], station_y[iSQ], inv_wavelength,
                time_int_sec, gha0_rad, dec0_rad, &du_dt, &dv_dt, &dw_dt);
    }

    for (int src_start = 0; src_start < num_sources; src_start += BLOCK_SINGLE)
    {
        /* Load parts of Jones matrix block into shared memory. */
        if (iSQ < num_stations && src_start + COL < num_sources)
        {
            OSKAR_LOAD_MATRIX(As[ROW][COL], jones, iSQ * num_sources + src_start + COL);
        }
        if (iSP < num_stations && src_start + ROW < num_sources)
        {
            OSKAR_LOAD_MATRIX(Bs[COL][ROW], jones, iSP * num_sources + src_start + COL);
        }

        /* Get per-source data. */
        int i = src_start + ROW;
        if (COL == 0 && i < num_sources)
        {
            ls[ROW] = source_l[i];
            ms[ROW] = source_m[i];
            ns[ROW] = source_n[i] - 1.0f;
        }

        /* Multiply the two sub-matrices together. */
        __syncthreads();
        if (iSP < num_stations && iSQ < num_stations)
        {
            for (int i = 0; i < BLOCK_SINGLE; ++i)
            {
                float r;
                if (src_start + i >= num_sources) break;

                /* Compute bandwidth- and time-smearing terms. */
                {
                    float l = ls[i];
                    float m = ms[i];
                    float n = ns[i];
                    r =  oskar_sinc_f(uu * l + vv * m + ww * n);
                    r *= oskar_sinc_f(du_dt * l + dv_dt * m + dw_dt * n);
                }

                /* Multiply Jones matrices. */
                float4c m1 = As[ROW][i];
                float4c m2 = Bs[i][COL];
//                float4c m1, m2;
//                OSKAR_LOAD_MATRIX(m1, jones, iSQ * num_sources + s);
//                OSKAR_LOAD_MATRIX(m2, jones, iSP * num_sources + s);
                oskar_multiply_complex_matrix_conjugate_transpose_in_place_f(
                        &m1, &m2);

                /* Multiply result by smearing term and accumulate. */
                sum.a.x += m1.a.x * r;
                sum.a.y += m1.a.y * r;
                sum.b.x += m1.b.x * r;
                sum.b.y += m1.b.y * r;
                sum.c.x += m1.c.x * r;
                sum.c.y += m1.c.y * r;
                sum.d.x += m1.d.x * r;
                sum.d.y += m1.d.y * r;
            }
        }
        __syncthreads();
    }

    /* Write result to global memory. */
    if (iSP > iSQ && iSP < num_stations && iSQ < num_stations)
    {
        int i = oskar_evaluate_baseline_index_inline(num_stations, iSP, iSQ);
        oskar_add_complex_matrix_in_place_f(&vis[i], &sum);
    }
}

/* Double precision. */
__global__
void oskar_correlate_point_time_smearing_new_cudak_d(const int num_sources,
        const int num_stations, const double4c* restrict jones,
        const double* restrict source_l, const double* restrict source_m,
        const double* restrict source_n, const double* restrict station_u,
        const double* restrict station_v, const double* restrict station_w,
        const double* restrict station_x, const double* restrict station_y,
        const double uv_min_lambda, const double uv_max_lambda,
        const double inv_wavelength, const double frac_bandwidth,
        const double time_int_sec, const double gha0_rad, const double dec0_rad,
        double4c* restrict vis)
{
//    __shared__ double4c As[BLOCK_DOUBLE][BLOCK_DOUBLE];
//    __shared__ double4c Bs[BLOCK_DOUBLE][BLOCK_DOUBLE];
    __shared__ double ls[BLOCK_DOUBLE], ms[BLOCK_DOUBLE], ns[BLOCK_DOUBLE];
    double uv_len, uu, vv, ww, uu2, vv2, uuvv, du_dt, dv_dt, dw_dt;

    /* Exit if block is in the wrong half of the visibility matrix. */
    if (blockIdx.y > blockIdx.x) return;

    /* Each thread block computes one sub-matrix of the visibility matrix.
     * Each thread computes visibilities for one baseline. */
    double4c sum;
    oskar_clear_complex_matrix_d(&sum);

    /* Get station indices. */
    const int iSP = blockIdx.x * BLOCK_DOUBLE + COL;
    const int iSQ = blockIdx.y * BLOCK_DOUBLE + ROW;

//    if (iSP >= num_stations || iSQ >= num_stations || iSP <= iSQ)
//        return;

    /* Evaluate per-baseline terms. */
    if (iSP < num_stations && iSQ < num_stations)
    {
        oskar_evaluate_baseline_terms_inline_d(station_u[iSP],
                station_u[iSQ], station_v[iSP], station_v[iSQ],
                station_w[iSP], station_w[iSQ], inv_wavelength,
                frac_bandwidth, &uv_len, &uu, &vv, &ww, &uu2, &vv2, &uuvv);
        oskar_evaluate_baseline_derivatives_inline_d(station_x[iSP],
                station_x[iSQ], station_y[iSP], station_y[iSQ], inv_wavelength,
                time_int_sec, gha0_rad, dec0_rad, &du_dt, &dv_dt, &dw_dt);
    }

    for (int src_start = 0; src_start < num_sources; src_start += BLOCK_DOUBLE)
    {
        /* Load parts of Jones matrix block into shared memory. */
//        if (iSQ < num_stations && src_start + COL < num_sources)
//        {
//            OSKAR_LOAD_MATRIX(As[ROW][COL], jones, iSQ * num_sources + src_start + COL);
//        }
//        if (iSP < num_stations && src_start + ROW < num_sources)
//        {
//            OSKAR_LOAD_MATRIX(Bs[COL][ROW], jones, iSP * num_sources + src_start + COL);
//        }

        /* Get per-source data. */
        int i = src_start + ROW;
        if (COL == 0 && i < num_sources)
        {
            ls[ROW] = source_l[i];
            ms[ROW] = source_m[i];
            ns[ROW] = source_n[i] - 1.0;
        }

        /* Multiply the two sub-matrices together. */
        __syncthreads();
        if (iSP < num_stations && iSQ < num_stations)
        {
            for (int i = 0; i < BLOCK_DOUBLE; ++i)
            {
                double r;
                const int s = src_start + i;
                if (src_start + i >= num_sources) break;

                /* Compute bandwidth- and time-smearing terms. */
                {
                    double l = ls[i];
                    double m = ms[i];
                    double n = ns[i];
                    r =  oskar_sinc_d(uu * l + vv * m + ww * n);
                    r *= oskar_sinc_d(du_dt * l + dv_dt * m + dw_dt * n);
                }

                /* Multiply Jones matrices. */
                double4c m1, m2;
                OSKAR_LOAD_MATRIX(m1, jones, iSQ * num_sources + s);
                OSKAR_LOAD_MATRIX(m2, jones, iSP * num_sources + s);
                oskar_multiply_complex_matrix_conjugate_transpose_in_place_d(
                        &m1, &m2);

                /* Multiply result by smearing term and accumulate. */
                sum.a.x += m1.a.x * r;
                sum.a.y += m1.a.y * r;
                sum.b.x += m1.b.x * r;
                sum.b.y += m1.b.y * r;
                sum.c.x += m1.c.x * r;
                sum.c.y += m1.c.y * r;
                sum.d.x += m1.d.x * r;
                sum.d.y += m1.d.y * r;
            }
        }
        __syncthreads();
    }

    /* Write result to global memory. */
    if (iSP > iSQ && iSP < num_stations && iSQ < num_stations)
    {
        int i = oskar_evaluate_baseline_index_inline(num_stations, iSP, iSQ);
        oskar_add_complex_matrix_in_place_d(&vis[i], &sum);
    }
}


#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_correlate_point_time_smearing_new_cuda_f(int num_sources,
        int num_stations, const float4c* d_jones,
        const float* d_source_l, const float* d_source_m,
        const float* d_source_n, const float* d_station_u,
        const float* d_station_v, const float* d_station_w,
        const float* d_station_x, const float* d_station_y,
        float uv_min_lambda, float uv_max_lambda, float inv_wavelength,
        float frac_bandwidth, float time_int_sec, float gha0_rad,
        float dec0_rad, float4c* d_vis)
{
    dim3 num_threads(BLOCK_SINGLE, BLOCK_SINGLE);
    dim3 num_blocks((num_stations + BLOCK_SINGLE - 1) / BLOCK_SINGLE,
            (num_stations + BLOCK_SINGLE - 1) / BLOCK_SINGLE);
    oskar_correlate_point_time_smearing_new_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads)
    (num_sources, num_stations, d_jones, d_source_l, d_source_m, d_source_n,
            d_station_u, d_station_v, d_station_w, d_station_x, d_station_y,
            uv_min_lambda, uv_max_lambda, inv_wavelength, frac_bandwidth,
            time_int_sec, gha0_rad, dec0_rad, d_vis);
}

/* Double precision. */
void oskar_correlate_point_time_smearing_new_cuda_d(int num_sources,
        int num_stations, const double4c* d_jones,
        const double* d_source_l, const double* d_source_m,
        const double* d_source_n, const double* d_station_u,
        const double* d_station_v, const double* d_station_w,
        const double* d_station_x, const double* d_station_y,
        double uv_min_lambda, double uv_max_lambda, double inv_wavelength,
        double frac_bandwidth, double time_int_sec, double gha0_rad,
        double dec0_rad, double4c* d_vis)
{
    dim3 num_threads(BLOCK_DOUBLE, BLOCK_DOUBLE);
    dim3 num_blocks((num_stations + BLOCK_DOUBLE - 1) / BLOCK_DOUBLE,
            (num_stations + BLOCK_DOUBLE - 1) / BLOCK_DOUBLE);
    oskar_correlate_point_time_smearing_new_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads)
    (num_sources, num_stations, d_jones, d_source_l, d_source_m, d_source_n,
            d_station_u, d_station_v, d_station_w, d_station_x, d_station_y,
            uv_min_lambda, uv_max_lambda, inv_wavelength, frac_bandwidth,
            time_int_sec, gha0_rad, dec0_rad, d_vis);
}

#ifdef __cplusplus
}
#endif
