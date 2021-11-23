/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station.h"
#include "math/oskar_random_gaussian.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_override_element_feed_angle(oskar_Station* station,
        int feed, unsigned int seed, double alpha_error_rad,
        double beta_error_rad, double gamma_error_rad, int* status)
{
    int i = 0;
    if (*status || !station) return;
    const int num = oskar_station_num_elements(station);
    if (oskar_station_has_child(station))
    {
        /* Recursive call to find the last level (i.e. the element data). */
        for (i = 0; i < num; ++i)
        {
            oskar_station_override_element_feed_angle(
                    oskar_station_child(station, i), feed, seed,
                    alpha_error_rad, beta_error_rad, gamma_error_rad, status);
        }
    }
    else
    {
        /* Override element data at last level. */
        double *a = 0, *b = 0, *c = 0, r[4];

        /* Get pointer to the X or Y element orientation data. */
        a = oskar_mem_double(
                oskar_station_element_euler_rad(station, feed, 0), status);
        b = oskar_mem_double(
                oskar_station_element_euler_rad(station, feed, 1), status);
        c = oskar_mem_double(
                oskar_station_element_euler_rad(station, feed, 2), status);
        const int id = oskar_station_unique_id(station);
        for (i = 0; i < num; ++i)
        {
            /* Generate random numbers from Gaussian distribution. */
            oskar_random_gaussian4(seed, i, id, 0, 0, r);
            r[0] *= alpha_error_rad;
            r[1] *= beta_error_rad;
            r[2] *= gamma_error_rad;

            /* Set the new angle. */
            a[i] += r[0];
            b[i] += r[1];
            c[i] += r[2];
        }
    }
}

#ifdef __cplusplus
}
#endif
