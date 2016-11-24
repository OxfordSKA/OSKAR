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

#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station.h"
#include "math/oskar_random_gaussian.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_override_element_gains(oskar_Station* s, unsigned int seed,
        double gain_mean, double gain_std, int* status)
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
            oskar_station_override_element_gains(oskar_station_child(s, i),
                    seed, gain_mean, gain_std, status);
        }
    }
    else
    {
        /* Override element data at last level. */
        double r[2];
        int type, id;
        type = oskar_station_precision(s);
        id = oskar_station_unique_id(s);
        if (gain_mean <= 0.0) gain_mean = 1.0;
        if (type == OSKAR_DOUBLE)
        {
            double* gain;
            gain = oskar_mem_double(s->element_gain, status);
            for (i = 0; i < s->num_elements; ++i)
            {
                oskar_random_gaussian2(seed, i, id, r);
                gain[i] = gain_mean + gain_std * r[0];
            }
        }
        else if (type == OSKAR_SINGLE)
        {
            float* gain;
            gain = oskar_mem_float(s->element_gain, status);
            for (i = 0; i < s->num_elements; ++i)
            {
                oskar_random_gaussian2(seed, i, id, r);
                gain[i] = gain_mean + gain_std * r[0];
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
