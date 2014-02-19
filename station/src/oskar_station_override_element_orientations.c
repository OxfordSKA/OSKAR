/*
 * Copyright (c) 2013-2014, The University of Oxford
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
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_override_element_orientations(oskar_Station* s,
        int x_pol, double orientation_error_rad, int* status)
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
    if (oskar_station_location(s) != OSKAR_LOCATION_CPU)
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
            oskar_station_override_element_orientations(
                    oskar_station_child(s, i), x_pol, orientation_error_rad,
                    status);
        }
    }
    else
    {
        /* Override element data at last level. */
        int type;
        double delta, angle;
        oskar_Mem *mem_cos, *mem_sin;

        /* Get pointers to the X or Y element orientation data. */
        mem_cos = x_pol ? s->cos_orientation_x : s->cos_orientation_y;
        mem_sin = x_pol ? s->sin_orientation_x : s->sin_orientation_y;
        type = oskar_station_precision(s);
        if (type == OSKAR_DOUBLE)
        {
            double *cos_x, *sin_x;
            cos_x = oskar_mem_double(mem_cos, status);
            sin_x = oskar_mem_double(mem_sin, status);
            for (i = 0; i < s->num_elements; ++i)
            {
                /* Generate random number from Gaussian distribution. */
                delta = orientation_error_rad * oskar_random_gaussian(0);

                /* Get the new angle. */
                angle = delta + atan2(sin_x[i], cos_x[i]);
                cos_x[i] = cos(angle);
                sin_x[i] = sin(angle);
            }
        }
        else if (type == OSKAR_SINGLE)
        {
            float *cos_x, *sin_x;
            cos_x = oskar_mem_float(mem_cos, status);
            sin_x = oskar_mem_float(mem_sin, status);
            for (i = 0; i < s->num_elements; ++i)
            {
                /* Generate random number from Gaussian distribution. */
                delta = orientation_error_rad * oskar_random_gaussian(0);

                /* Get the new angle. */
                angle = delta + atan2(sin_x[i], cos_x[i]);
                cos_x[i] = (float) cos(angle);
                sin_x[i] = (float) sin(angle);
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
