/*
 * Copyright (c) 2014-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/oskar_sky.h"

#include "math/oskar_random_gaussian.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_override_polarisation(oskar_Sky* sky, double mean_pol_fraction,
        double std_pol_fraction, double mean_pol_angle_rad,
        double std_pol_angle_rad, int seed, int* status)
{
    if (*status) return;

    /* Skip if not required. */
    if (mean_pol_fraction < 0.0) return;

    /* Get the data location and type. */
    const int location = oskar_sky_mem_location(sky);
    const int type = oskar_sky_precision(sky);
    const int num_sources = oskar_sky_num_sources(sky);

    if (location == OSKAR_CPU)
    {
        int i = 0;
        double r[2];

        if (type == OSKAR_DOUBLE)
        {
            double *Q_ = 0, *U_ = 0;
            const double *I_ = 0;
            I_ = oskar_mem_double_const(oskar_sky_I_const(sky), status);
            Q_ = oskar_mem_double(oskar_sky_Q(sky), status);
            U_ = oskar_mem_double(oskar_sky_U(sky), status);
            for (i = 0; i < num_sources; ++i)
            {
                /* Generate polarisation angle and polarisation fraction. */
                oskar_random_gaussian2(seed, i, 0, r);
                r[0] = 2.0 * (std_pol_angle_rad * r[0] + mean_pol_angle_rad);
                r[1] = std_pol_fraction * r[1] + mean_pol_fraction;
                if (r[1] > 1.0)
                {
                    r[1] = 1.0; /* Clamp pol fraction. */
                }
                else if (r[1] < 0.0)
                {
                    r[1] = 0.0;
                }

                /* Generate Stokes Q and U values. */
                Q_[i] = r[1] * I_[i] * cos(r[0]);
                U_[i] = r[1] * I_[i] * sin(r[0]);
            }
        }
        else if (type == OSKAR_SINGLE)
        {
            float *Q_ = 0, *U_ = 0;
            const float *I_ = 0;
            I_ = oskar_mem_float_const(oskar_sky_I_const(sky), status);
            Q_ = oskar_mem_float(oskar_sky_Q(sky), status);
            U_ = oskar_mem_float(oskar_sky_U(sky), status);
            for (i = 0; i < num_sources; ++i)
            {
                /* Generate polarisation angle and polarisation fraction. */
                oskar_random_gaussian2(seed, i, 0, r);
                r[0] = 2.0 * (std_pol_angle_rad * r[0] + mean_pol_angle_rad);
                r[1] = std_pol_fraction * r[1] + mean_pol_fraction;
                if (r[1] > 1.0)
                {
                    r[1] = 1.0; /* Clamp pol fraction. */
                }
                else if (r[1] < 0.0)
                {
                    r[1] = 0.0;
                }

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
