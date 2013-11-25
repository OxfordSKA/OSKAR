/*
 * Copyright (c) 2013, The University of Oxford
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
#include <oskar_accumulate_baseline_visibility_for_source.h>
#include <oskar_correlate_point_time_smearing_omp.h>
#include <oskar_sinc.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

#define OMEGA_EARTH  7.272205217e-5  /* radians/sec */
#define OMEGA_EARTHf 7.272205217e-5f /* radians/sec */


#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_correlate_point_time_smearing_omp_f(int num_sources,
        int num_stations, const float4c* jones, const float* source_I,
        const float* source_Q, const float* source_U, const float* source_V,
        const float* source_l, const float* source_m, const float* source_n,
        const float* station_u, const float* station_v,
        const float* station_x, const float* station_y, float inv_wavelength,
        float frac_bandwidth, float time_int_sec, float gha0_rad,
        float dec0_rad, float4c* vis)
{
    int station_q;

    /* Loop over stations. */
#pragma omp parallel for private(station_q) schedule(dynamic, 1)
    for (station_q = 0; station_q < num_stations; ++station_q)
    {
        int station_p, i;
        const float4c *sp, *sq;

        /* Pointer to source vector for station q. */
        sq = &jones[station_q * num_sources];

        /* Loop over baselines for this station. */
        for (station_p = station_q + 1; station_p < num_stations; ++station_p)
        {
            float uu, vv, du_dt, dv_dt, dw_dt, factor;
            float4c sum, guard;
            sum.a.x = 0.0f;
            sum.a.y = 0.0f;
            sum.b.x = 0.0f;
            sum.b.y = 0.0f;
            sum.c.x = 0.0f;
            sum.c.y = 0.0f;
            sum.d.x = 0.0f;
            sum.d.y = 0.0f;
            guard.a.x = 0.0f;
            guard.a.y = 0.0f;
            guard.b.x = 0.0f;
            guard.b.y = 0.0f;
            guard.c.x = 0.0f;
            guard.c.y = 0.0f;
            guard.d.x = 0.0f;
            guard.d.y = 0.0f;

            /* Pointer to source vector for station p. */
            sp = &jones[station_p * num_sources];

            /* Baseline lengths. */
            factor = M_PIf * inv_wavelength;
            uu = (station_u[station_p] - station_u[station_q]) * factor;
            vv = (station_v[station_p] - station_v[station_q]) * factor;

            /* Modify the baseline distance to include the common components
             * of the bandwidth smearing term. */
            uu *= frac_bandwidth;
            vv *= frac_bandwidth;

            /* Compute the derivatives for time-average smearing. */
            {
                float xx, yy, rot_angle, temp;
                float sin_HA, cos_HA, sin_Dec, cos_Dec;
                sin_HA = sinf(gha0_rad);
                cos_HA = cosf(gha0_rad);
                sin_Dec = sinf(dec0_rad);
                cos_Dec = cosf(dec0_rad);
                xx = (station_x[station_p] - station_x[station_q]) * factor;
                yy = (station_y[station_p] - station_y[station_q]) * factor;
                rot_angle = OMEGA_EARTHf * time_int_sec;
                temp = (xx * sin_HA + yy * cos_HA) * rot_angle;
                du_dt = (xx * cos_HA - yy * sin_HA) * rot_angle;
                dv_dt = temp * sin_Dec;
                dw_dt = -temp * cos_Dec;
            }

            /* Loop over sources. */
            for (i = 0; i < num_sources; ++i)
            {
                float l, m, n, rb, rt;

                /* Get source direction cosines. */
                l = source_l[i];
                m = source_m[i];
                n = source_n[i];

                /* Compute bandwidth- and time-smearing terms. */
                rb = oskar_sinc_f(uu * l + vv * m);
                rt = oskar_sinc_f(du_dt * l + dv_dt * m +
                        dw_dt * (fabsf(n) - 1.0f));
                rb *= rt;

                /* Accumulate baseline visibility response for source. */
                oskar_accumulate_baseline_visibility_for_source_f(&sum, i,
                        source_I, source_Q, source_U, source_V,
                        sp, sq, rb, &guard);
            }

            /* Determine 1D visibility index. */
            i = station_q*(num_stations-1) - (station_q-1)*station_q/2 +
                    station_p - station_q - 1;

            /* Add result to the baseline visibility. */
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
}

/* Double precision. */
void oskar_correlate_point_time_smearing_omp_d(int num_sources,
        int num_stations, const double4c* jones, const double* source_I,
        const double* source_Q, const double* source_U, const double* source_V,
        const double* source_l, const double* source_m, const double* source_n,
        const double* station_u, const double* station_v,
        const double* station_x, const double* station_y, double inv_wavelength,
        double frac_bandwidth, double time_int_sec, double gha0_rad,
        double dec0_rad, double4c* vis)
{
    int station_q;

    /* Loop over stations. */
#pragma omp parallel for private(station_q) schedule(dynamic, 1)
    for (station_q = 0; station_q < num_stations; ++station_q)
    {
        int station_p, i;
        const double4c *sp, *sq;

        /* Pointer to source vector for station q. */
        sq = &jones[station_q * num_sources];

        /* Loop over baselines for this station. */
        for (station_p = station_q + 1; station_p < num_stations; ++station_p)
        {
            double uu, vv, du_dt, dv_dt, dw_dt, factor;
            double4c sum;
            sum.a.x = 0.0;
            sum.a.y = 0.0;
            sum.b.x = 0.0;
            sum.b.y = 0.0;
            sum.c.x = 0.0;
            sum.c.y = 0.0;
            sum.d.x = 0.0;
            sum.d.y = 0.0;

            /* Pointer to source vector for station p. */
            sp = &jones[station_p * num_sources];

            /* Baseline lengths. */
            factor = M_PI * inv_wavelength;
            uu = (station_u[station_p] - station_u[station_q]) * factor;
            vv = (station_v[station_p] - station_v[station_q]) * factor;

            /* Modify the baseline distance to include the common components
             * of the bandwidth smearing term. */
            uu *= frac_bandwidth;
            vv *= frac_bandwidth;

            /* Compute the derivatives for time-average smearing. */
            {
                double xx, yy, rot_angle, temp;
                double sin_HA, cos_HA, sin_Dec, cos_Dec;
                sin_HA = sin(gha0_rad);
                cos_HA = cos(gha0_rad);
                sin_Dec = sin(dec0_rad);
                cos_Dec = cos(dec0_rad);
                xx = (station_x[station_p] - station_x[station_q]) * factor;
                yy = (station_y[station_p] - station_y[station_q]) * factor;
                rot_angle = OMEGA_EARTH * time_int_sec;
                temp = (xx * sin_HA + yy * cos_HA) * rot_angle;
                du_dt = (xx * cos_HA - yy * sin_HA) * rot_angle;
                dv_dt = temp * sin_Dec;
                dw_dt = -temp * cos_Dec;
            }

            /* Loop over sources. */
            for (i = 0; i < num_sources; ++i)
            {
                double l, m, n, rb, rt;

                /* Get source direction cosines. */
                l = source_l[i];
                m = source_m[i];
                n = source_n[i];

                /* Compute bandwidth- and time-smearing terms. */
                rb = oskar_sinc_d(uu * l + vv * m);
                rt = oskar_sinc_d(du_dt * l + dv_dt * m +
                        dw_dt * (fabs(n) - 1.0));
                rb *= rt;

                /* Accumulate baseline visibility response for source. */
                oskar_accumulate_baseline_visibility_for_source_d(&sum, i,
                        source_I, source_Q, source_U, source_V,
                        sp, sq, rb);
            }

            /* Determine 1D visibility index. */
            i = station_q*(num_stations-1) - (station_q-1)*station_q/2 +
                    station_p - station_q - 1;

            /* Add result to the baseline visibility. */
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
}

#ifdef __cplusplus
}
#endif
