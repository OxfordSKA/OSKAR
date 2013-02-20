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

#include "station/oskar_station_model_override_element_xy_position_errors.h"
#include "station/oskar_station_model_type.h"
#include "station/oskar_station_model_location.h"
#include "math/oskar_random_gaussian.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_model_override_element_xy_position_errors(
        oskar_StationModel* s, double position_error_xy_m, int* status)
{
    int i;

    /* Check all inputs. */
    if (!s || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check location. */
    if (oskar_station_model_location(s) != OSKAR_LOCATION_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    /* Check coordinate units are in metres. */
    if (s->coord_units != OSKAR_METRES)
    {
        *status = OSKAR_ERR_BAD_UNITS;
        return;
    }

    /* Check if there are child stations. */
    if (s->child)
    {
        /* Recursive call to find the last level (i.e. the element data). */
        for (i = 0; i < s->num_elements; ++i)
        {
            oskar_station_model_override_element_xy_position_errors(
                    &s->child[i], position_error_xy_m, status);
        }
    }
    else
    {
        /* Override element data at last level. */
        int type;
        double delta_x, delta_y;
        type = oskar_station_model_type(s);
        if (type == OSKAR_DOUBLE)
        {
            double *xs, *ys, *xw, *yw;
            xs = (double*)(s->x_signal.data);
            ys = (double*)(s->y_signal.data);
            xw = (double*)(s->x_weights.data);
            yw = (double*)(s->y_weights.data);
            for (i = 0; i < s->num_elements; ++i)
            {
                /* Generate random numbers from Gaussian distribution. */
                delta_x = oskar_random_gaussian(&delta_y);
                delta_x *= position_error_xy_m;
                delta_y *= position_error_xy_m;
                xs[i] = xw[i] + delta_x;
                ys[i] = yw[i] + delta_y;
            }
        }
        else if (type == OSKAR_SINGLE)
        {
            float *xs, *ys, *xw, *yw;
            xs = (float*)(s->x_signal.data);
            ys = (float*)(s->y_signal.data);
            xw = (float*)(s->x_weights.data);
            yw = (float*)(s->y_weights.data);
            for (i = 0; i < s->num_elements; ++i)
            {
                /* Generate random numbers from Gaussian distribution. */
                delta_x = oskar_random_gaussian(&delta_y);
                delta_x *= position_error_xy_m;
                delta_y *= position_error_xy_m;
                xs[i] = xw[i] + delta_x;
                ys[i] = yw[i] + delta_y;
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
