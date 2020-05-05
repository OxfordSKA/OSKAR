/*
 * Copyright (c) 2013-2020, The University of Oxford
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

#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station.h"
#include "math/oskar_random_gaussian.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_override_element_xy_position_errors(oskar_Station* station,
        int feed, unsigned int seed, double position_error_xy_m, int* status)
{
    int i;
    if (*status || !station) return;
    const int loc = oskar_station_mem_location(station);
    if (loc != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }
    const int num = station->num_elements;
    if (oskar_station_has_child(station))
    {
        /* Recursive call to find the last level (i.e. the element data). */
        for (i = 0; i < num; ++i)
        {
            oskar_station_override_element_xy_position_errors(
                    oskar_station_child(station, i),
                    feed, seed, position_error_xy_m, status);
        }
    }
    else
    {
        /* Override element data at last level. */
        oskar_Mem *ptr_true[2], *ptr_meas[2];
        double r[2];
        int dim;
        const int type = oskar_station_precision(station);
        const int id = oskar_station_unique_id(station);
        for (dim = 0; dim < 2; dim++)
        {
            ptr_true[dim] = station->element_true_enu_metres[feed][dim];
            if (!ptr_true[dim])
            {
                station->element_true_enu_metres[feed][dim] =
                        oskar_mem_create(type, loc, num, status);
                ptr_true[dim] = station->element_true_enu_metres[feed][dim];
            }
            ptr_meas[dim] = station->element_measured_enu_metres[feed][dim];
            if (!ptr_meas[dim])
            {
                station->element_measured_enu_metres[feed][dim] =
                        oskar_mem_create(type, loc, num, status);
                ptr_meas[dim] = station->element_measured_enu_metres[feed][dim];
            }
        }
        if (type == OSKAR_DOUBLE)
        {
            double *xs, *ys, *xw, *yw;
            xs = oskar_mem_double(ptr_true[0], status);
            ys = oskar_mem_double(ptr_true[1], status);
            xw = oskar_mem_double(ptr_meas[0], status);
            yw = oskar_mem_double(ptr_meas[1], status);
            for (i = 0; i < num; ++i)
            {
                /* Generate random numbers from Gaussian distribution. */
                oskar_random_gaussian2(seed, i, id, r);
                r[0] *= position_error_xy_m;
                r[1] *= position_error_xy_m;
                xs[i] = xw[i] + r[0];
                ys[i] = yw[i] + r[1];
            }
        }
        else if (type == OSKAR_SINGLE)
        {
            float *xs, *ys, *xw, *yw;
            xs = oskar_mem_float(ptr_true[0], status);
            ys = oskar_mem_float(ptr_true[1], status);
            xw = oskar_mem_float(ptr_meas[0], status);
            yw = oskar_mem_float(ptr_meas[1], status);
            for (i = 0; i < num; ++i)
            {
                /* Generate random numbers from Gaussian distribution. */
                oskar_random_gaussian2(seed, i, id, r);
                r[0] *= position_error_xy_m;
                r[1] *= position_error_xy_m;
                xs[i] = xw[i] + r[0];
                ys[i] = yw[i] + r[1];
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
