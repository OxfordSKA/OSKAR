/*
 * Copyright (c) 2014-2015, The University of Oxford
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

#include <oskar_sky.h>

#include <oskar_random_gaussian.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_override_polarisation(oskar_Sky* sky, double mean_pol_fraction,
        double std_pol_fraction, double mean_pol_angle_rad,
        double std_pol_angle_rad, int seed, int* status)
{
    int type, location, num_sources;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Skip if not required. */
    if (mean_pol_fraction < 0.0)
        return;

    /* Get the data location and type. */
    location = oskar_sky_mem_location(sky);
    type = oskar_sky_precision(sky);
    num_sources = oskar_sky_num_sources(sky);

    if (location == OSKAR_CPU)
    {
        int i;
        double r[2];

        if (type == OSKAR_DOUBLE)
        {
            double *Q_, *U_;
            const double *I_;
            I_ = oskar_mem_double_const(oskar_sky_I_const(sky), status);
            Q_ = oskar_mem_double(oskar_sky_Q(sky), status);
            U_ = oskar_mem_double(oskar_sky_U(sky), status);
            for (i = 0; i < num_sources; ++i)
            {
                /* Generate polarisation angle and polarisation fraction. */
                oskar_random_gaussian2(seed, i, 0, r);
                r[0] = 2.0 * (std_pol_angle_rad * r[0] + mean_pol_angle_rad);
                r[1] = std_pol_fraction * r[1] + mean_pol_fraction;
                if (r[1] > 1.0) r[1] = 1.0; /* Clamp pol fraction. */
                else if (r[1] < 0.0) r[1] = 0.0;

                /* Generate Stokes Q and U values. */
                Q_[i] = r[1] * I_[i] * cos(r[0]);
                U_[i] = r[1] * I_[i] * sin(r[0]);
            }
        }
        else if (type == OSKAR_SINGLE)
        {
            float *Q_, *U_;
            const float *I_;
            I_ = oskar_mem_float_const(oskar_sky_I_const(sky), status);
            Q_ = oskar_mem_float(oskar_sky_Q(sky), status);
            U_ = oskar_mem_float(oskar_sky_U(sky), status);
            for (i = 0; i < num_sources; ++i)
            {
                /* Generate polarisation angle and polarisation fraction. */
                oskar_random_gaussian2(seed, i, 0, r);
                r[0] = 2.0 * (std_pol_angle_rad * r[0] + mean_pol_angle_rad);
                r[1] = std_pol_fraction * r[1] + mean_pol_fraction;
                if (r[1] > 1.0) r[1] = 1.0; /* Clamp pol fraction. */
                else if (r[1] < 0.0) r[1] = 0.0;

                /* Generate Stokes Q and U values. */
                Q_[i] = r[1] * I_[i] * cos(r[0]);
                U_[i] = r[1] * I_[i] * sin(r[0]);
            }
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
    else
    {
        *status = OSKAR_ERR_BAD_LOCATION;
    }
}

#ifdef __cplusplus
}
#endif
