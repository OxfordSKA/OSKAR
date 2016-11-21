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

#include <private_station.h>
#include <oskar_station.h>
#include <oskar_random_gaussian.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_override_element_xy_position_errors(oskar_Station* s,
        unsigned int seed, double position_error_xy_m, int* status)
{
    int i;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check location. */
    if (oskar_station_mem_location(s) != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    /* Check if there are child stations. */
    if (oskar_station_has_child(s))
    {
        /* Recursive call to find the last level (i.e. the element data). */
        for (i = 0; i < s->num_elements; ++i)
        {
            oskar_station_override_element_xy_position_errors(
                    oskar_station_child(s, i), seed, position_error_xy_m,
                    status);
        }
    }
    else
    {
        /* Override element data at last level. */
        double r[2];
        int type, id;
        type = oskar_station_precision(s);
        id = oskar_station_unique_id(s);
        if (type == OSKAR_DOUBLE)
        {
            double *xs, *ys, *xw, *yw;
            xs = oskar_mem_double(s->element_true_x_enu_metres, status);
            ys = oskar_mem_double(s->element_true_y_enu_metres, status);
            xw = oskar_mem_double(s->element_measured_x_enu_metres, status);
            yw = oskar_mem_double(s->element_measured_y_enu_metres, status);
            for (i = 0; i < s->num_elements; ++i)
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
            xs = oskar_mem_float(s->element_true_x_enu_metres, status);
            ys = oskar_mem_float(s->element_true_y_enu_metres, status);
            xw = oskar_mem_float(s->element_measured_x_enu_metres, status);
            yw = oskar_mem_float(s->element_measured_y_enu_metres, status);
            for (i = 0; i < s->num_elements; ++i)
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
