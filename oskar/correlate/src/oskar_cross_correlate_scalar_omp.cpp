/*
 * Copyright (c) 2014-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "correlate/define_correlate_utils.h"
#include "correlate/oskar_cross_correlate_scalar_omp.h"
#include "math/define_multiply.h"
#include "math/oskar_kahan_sum.h"
#include "utility/oskar_kernel_macros.h"
#include "utility/oskar_vector_types.h"
#include <stdio.h>

template<typename T1, typename T2>
struct oskar_IsSame
{
    enum { value = false }; // oskar_IsSame represents a bool.
    typedef oskar_IsSame<T1,T2> type; // to qualify as a metafunction.
};

template<typename T>
struct oskar_IsSame<T,T>
{
    enum { value = true };
    typedef oskar_IsSame<T,T> type;
};

template
<
// Compile-time parameters.
bool BANDWIDTH_SMEARING, bool TIME_SMEARING, bool GAUSSIAN,
typename REAL, typename REAL2
>
void oskar_xcorr_scalar_omp(
        const int                   num_sources,
        const int                   num_stations,
        const int                   offset_out,
        const REAL2* const RESTRICT jones,
        const REAL*  const RESTRICT source_I,
        const REAL*  const RESTRICT source_l,
        const REAL*  const RESTRICT source_m,
        const REAL*  const RESTRICT source_n,
        const REAL*  const RESTRICT source_a,
        const REAL*  const RESTRICT source_b,
        const REAL*  const RESTRICT source_c,
        const REAL*  const RESTRICT station_u,
        const REAL*  const RESTRICT station_v,
        const REAL*  const RESTRICT station_w,
        const REAL*  const RESTRICT station_x,
        const REAL*  const RESTRICT station_y,
        const REAL                  uv_min_lambda,
        const REAL                  uv_max_lambda,
        const REAL                  inv_wavelength,
        const REAL                  frac_bandwidth,
        const REAL                  time_int_sec,
        const REAL                  gha0_rad,
        const REAL                  dec0_rad,
        REAL2*             RESTRICT vis)
{
    // Loop over stations.
#pragma omp parallel for schedule(dynamic, 1)
    for (int SQ = 0; SQ < num_stations; ++SQ)
    {
        // Pointer to source vector for station q.
        const REAL2* const station_q = &jones[SQ * num_sources];

        // Loop over baselines for this station.
        for (int SP = SQ + 1; SP < num_stations; ++SP)
        {
            REAL uv_len, uu, vv, ww, uu2, vv2, uuvv, du, dv, dw;
            REAL2 t1, t2, sum, guard;
            sum.x = sum.y = (REAL) 0;
            if (oskar_IsSame<REAL, float>::value)
            {
                guard.x = guard.y = (REAL) 0;
            }

            // Pointer to source vector for station p.
            const REAL2* const station_p = &jones[SP * num_sources];

            // Get common baseline values.
            OSKAR_BASELINE_TERMS(REAL, station_u[SP], station_u[SQ],
                    station_v[SP], station_v[SQ], station_w[SP], station_w[SQ],
                    uu, vv, ww, uu2, vv2, uuvv, uv_len);

            // Apply the baseline length filter.
            if (uv_len < uv_min_lambda || uv_len > uv_max_lambda) continue;

            // Compute the deltas for time-average smearing.
            if (TIME_SMEARING)
                OSKAR_BASELINE_DELTAS(REAL, station_x[SP], station_x[SQ],
                        station_y[SP], station_y[SQ], du, dv, dw);

            // Loop over sources.
            for (int i = 0; i < num_sources; ++i)
            {
                REAL smearing;
                if (GAUSSIAN)
                {
                    const REAL t = source_a[i] * uu2 + source_b[i] * uuvv +
                            source_c[i] * vv2;
                    smearing = exp((REAL) -t);
                }
                else
                {
                    smearing = (REAL) 1;
                }
                smearing *= source_I[i];
                if (BANDWIDTH_SMEARING || TIME_SMEARING)
                {
                    const REAL l = source_l[i];
                    const REAL m = source_m[i];
                    const REAL n = source_n[i] - (REAL) 1;
                    if (BANDWIDTH_SMEARING)
                    {
                        const REAL t = uu * l + vv * m + ww * n;
                        smearing *= OSKAR_SINC(REAL, t);
                    }
                    if (TIME_SMEARING)
                    {
                        const REAL t = du * l + dv * m + dw * n;
                        smearing *= OSKAR_SINC(REAL, t);
                    }
                }

                // Multiply Jones scalars.
                t1 = station_p[i];
                t2 = station_q[i];
                OSKAR_MUL_COMPLEX_CONJUGATE_IN_PLACE(REAL2, t1, t2)

                // Multiply result by smearing term and accumulate.
                if (oskar_IsSame<REAL, float>::value)
                {
                    OSKAR_KAHAN_SUM_MULTIPLY_COMPLEX(
                            REAL, sum, t1, smearing, guard)
                }
                else
                {
                    sum.x += t1.x * smearing;
                    sum.y += t1.y * smearing;
                }
            }

            // Add result to the baseline visibility.
            int i = OSKAR_BASELINE_INDEX(num_stations, SP, SQ) + offset_out;
            vis[i].x += sum.x;
            vis[i].y += sum.y;
        }
    }
}

#define XCORR_KERNEL(BS, TS, GAUSSIAN, REAL, REAL2)                         \
        oskar_xcorr_scalar_omp<BS, TS, GAUSSIAN, REAL, REAL2>               \
        (num_sources, num_stations, offset_out, d_jones, d_I, d_l, d_m, d_n,\
                d_a, d_b, d_c, d_station_u, d_station_v, d_station_w,       \
                d_station_x, d_station_y, uv_min_lambda, uv_max_lambda,     \
                inv_wavelength, frac_bandwidth, time_int_sec,               \
                gha0_rad, dec0_rad, d_vis);

#define XCORR_SELECT(GAUSSIAN, REAL, REAL2)                                 \
        if (frac_bandwidth == (REAL)0 && time_int_sec == (REAL)0)           \
            XCORR_KERNEL(false, false, GAUSSIAN, REAL, REAL2)               \
        else if (frac_bandwidth != (REAL)0 && time_int_sec == (REAL)0)      \
            XCORR_KERNEL(true, false, GAUSSIAN, REAL, REAL2)                \
        else if (frac_bandwidth == (REAL)0 && time_int_sec != (REAL)0)      \
            XCORR_KERNEL(false, true, GAUSSIAN, REAL, REAL2)                \
        else if (frac_bandwidth != (REAL)0 && time_int_sec != (REAL)0)      \
            XCORR_KERNEL(true, true, GAUSSIAN, REAL, REAL2)

void oskar_cross_correlate_scalar_point_omp_f(
        int num_sources, int num_stations, int offset_out,
        const float2* d_jones, const float* d_I, const float* d_l,
        const float* d_m, const float* d_n,
        const float* d_station_u, const float* d_station_v,
        const float* d_station_w, const float* d_station_x,
        const float* d_station_y, float uv_min_lambda, float uv_max_lambda,
        float inv_wavelength, float frac_bandwidth, const float time_int_sec,
        const float gha0_rad, const float dec0_rad, float2* d_vis)
{
    const float *d_a = 0, *d_b = 0, *d_c = 0;
    XCORR_SELECT(false, float, float2)
}

void oskar_cross_correlate_scalar_point_omp_d(
        int num_sources, int num_stations, int offset_out,
        const double2* d_jones, const double* d_I, const double* d_l,
        const double* d_m, const double* d_n,
        const double* d_station_u, const double* d_station_v,
        const double* d_station_w, const double* d_station_x,
        const double* d_station_y, double uv_min_lambda, double uv_max_lambda,
        double inv_wavelength, double frac_bandwidth, const double time_int_sec,
        const double gha0_rad, const double dec0_rad, double2* d_vis)
{
    const double *d_a = 0, *d_b = 0, *d_c = 0;
    XCORR_SELECT(false, double, double2)
}

void oskar_cross_correlate_scalar_gaussian_omp_f(
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
    XCORR_SELECT(true, float, float2)
}

void oskar_cross_correlate_scalar_gaussian_omp_d(
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
    XCORR_SELECT(true, double, double2)
}
