/*
 * Copyright (c) 2013-2015, The University of Oxford
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

#include <math.h>
#include <private_correlate_functions_inline.h>
#include <oskar_cross_correlate_point_time_smearing_omp.h>
#include <oskar_add_inline.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_cross_correlate_point_time_smearing_omp_f(int num_sources,
        int num_stations, const float4c* jones, const float* source_I,
        const float* source_Q, const float* source_U, const float* source_V,
        const float* source_l, const float* source_m, const float* source_n,
        const float* station_u, const float* station_v,
        const float* station_w, const float* station_x,
        const float* station_y, float uv_min_lambda, float uv_max_lambda,
        float inv_wavelength, float frac_bandwidth, float time_int_sec,
        float gha0_rad, float dec0_rad, float4c* vis)
{
    int SQ;

    /* Loop over stations. */
#pragma omp parallel for private(SQ) schedule(dynamic, 1)
    for (SQ = 0; SQ < num_stations; ++SQ)
    {
        int SP, i;
        const float4c *station_p, *station_q;

        /* Pointer to source vector for station q. */
        station_q = &jones[SQ * num_sources];

        /* Loop over baselines for this station. */
        for (SP = SQ + 1; SP < num_stations; ++SP)
        {
            float uv_len, uu, vv, ww, uu2, vv2, uuvv, du_dt, dv_dt, dw_dt;
            float4c sum, guard;
            oskar_clear_complex_matrix_f(&sum);
            oskar_clear_complex_matrix_f(&guard);

            /* Pointer to source vector for station p. */
            station_p = &jones[SP * num_sources];

            /* Get common baseline values. */
            oskar_evaluate_baseline_terms_inline_f(station_u[SP],
                    station_u[SQ], station_v[SP], station_v[SQ],
                    station_w[SP], station_w[SQ], inv_wavelength,
                    frac_bandwidth, &uv_len, &uu, &vv, &ww, &uu2, &vv2, &uuvv);

            /* Apply the baseline length filter. */
            if (uv_len < uv_min_lambda || uv_len > uv_max_lambda)
                continue;

            /* Compute the derivatives for time-average smearing. */
            oskar_evaluate_baseline_derivatives_inline_f(station_x[SP],
                    station_x[SQ], station_y[SP], station_y[SQ],
                    inv_wavelength, time_int_sec, gha0_rad, dec0_rad,
                    &du_dt, &dv_dt, &dw_dt);

            /* Loop over sources. */
            for (i = 0; i < num_sources; ++i)
            {
                float l, m, n, r1, r2;

                /* Get source direction cosines. */
                l = source_l[i];
                m = source_m[i];
                n = source_n[i];

                /* Compute bandwidth- and time-smearing terms. */
                r1 = oskar_sinc_f(uu * l + vv * m + ww * (n - 1.0f));
                r2 = oskar_evaluate_time_smearing_f(du_dt, dv_dt, dw_dt,
                        l, m, n);
                r1 *= r2;

                /* Accumulate baseline visibility response for source. */
                oskar_accumulate_baseline_visibility_for_source_inline_f(&sum,
                        i, source_I, source_Q, source_U, source_V,
                        station_p, station_q, r1, &guard);
            }

            /* Add result to the baseline visibility. */
            i = oskar_evaluate_baseline_index_inline(num_stations, SP, SQ);
            oskar_add_complex_matrix_in_place_f(&vis[i], &sum);
        }
    }
}

/* Double precision. */
void oskar_cross_correlate_point_time_smearing_omp_d(int num_sources,
        int num_stations, const double4c* jones, const double* source_I,
        const double* source_Q, const double* source_U, const double* source_V,
        const double* source_l, const double* source_m, const double* source_n,
        const double* station_u, const double* station_v,
        const double* station_w, const double* station_x,
        const double* station_y, double uv_min_lambda, double uv_max_lambda,
        double inv_wavelength, double frac_bandwidth, double time_int_sec,
        double gha0_rad, double dec0_rad, double4c* vis)
{
    int SQ;

    /* Loop over stations. */
#pragma omp parallel for private(SQ) schedule(dynamic, 1)
    for (SQ = 0; SQ < num_stations; ++SQ)
    {
        int SP, i;
        const double4c *station_p, *station_q;

        /* Pointer to source vector for station q. */
        station_q = &jones[SQ * num_sources];

        /* Loop over baselines for this station. */
        for (SP = SQ + 1; SP < num_stations; ++SP)
        {
            double uv_len, uu, vv, ww, uu2, vv2, uuvv, du_dt, dv_dt, dw_dt;
            double4c sum;
            oskar_clear_complex_matrix_d(&sum);

            /* Pointer to source vector for station p. */
            station_p = &jones[SP * num_sources];

            /* Get common baseline values. */
            oskar_evaluate_baseline_terms_inline_d(station_u[SP],
                    station_u[SQ], station_v[SP], station_v[SQ],
                    station_w[SP], station_w[SQ], inv_wavelength,
                    frac_bandwidth, &uv_len, &uu, &vv, &ww, &uu2, &vv2, &uuvv);

            /* Apply the baseline length filter. */
            if (uv_len < uv_min_lambda || uv_len > uv_max_lambda)
                continue;

            /* Compute the derivatives for time-average smearing. */
            oskar_evaluate_baseline_derivatives_inline_d(station_x[SP],
                    station_x[SQ], station_y[SP], station_y[SQ],
                    inv_wavelength, time_int_sec, gha0_rad, dec0_rad,
                    &du_dt, &dv_dt, &dw_dt);

            /* Loop over sources. */
            for (i = 0; i < num_sources; ++i)
            {
                double l, m, n, r1, r2;

                /* Get source direction cosines. */
                l = source_l[i];
                m = source_m[i];
                n = source_n[i];

                /* Compute bandwidth- and time-smearing terms. */
                r1 = oskar_sinc_d(uu * l + vv * m + ww * (n - 1.0));
                r2 = oskar_evaluate_time_smearing_d(du_dt, dv_dt, dw_dt,
                        l, m, n);
                r1 *= r2;

                /* Accumulate baseline visibility response for source. */
                oskar_accumulate_baseline_visibility_for_source_inline_d(&sum,
                        i, source_I, source_Q, source_U, source_V,
                        station_p, station_q, r1);
            }

            /* Add result to the baseline visibility. */
            i = oskar_evaluate_baseline_index_inline(num_stations, SP, SQ);
            oskar_add_complex_matrix_in_place_d(&vis[i], &sum);
        }
    }
}

#ifdef __cplusplus
}
#endif
