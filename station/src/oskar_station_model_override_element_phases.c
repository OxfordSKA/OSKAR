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

#include "station/oskar_station_model_override_element_phases.h"
#include "station/oskar_station_model_type.h"
#include "station/oskar_station_model_location.h"
#include "math/oskar_random_gaussian.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_model_override_element_phases(oskar_StationModel* s,
        double phase_std, int* status)
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

    /* Check if there are child stations. */
    if (s->child)
    {
        /* Recursive call to find the last level (i.e. the element data). */
        for (i = 0; i < s->num_elements; ++i)
        {
            oskar_station_model_override_element_phases(&s->child[i],
                    phase_std, status);
        }
    }
    else
    {
        /* Override element data at last level. */
        int type;
        type = oskar_station_model_type(s);
        if (type == OSKAR_DOUBLE)
        {
            double* phase;
            phase = (double*)(s->phase_offset.data);
            for (i = 0; i < s->num_elements; ++i)
                phase[i] = phase_std * oskar_random_gaussian(0);
        }
        else if (type == OSKAR_SINGLE)
        {
            float* phase;
            phase = (float*)(s->phase_offset.data);
            for (i = 0; i < s->num_elements; ++i)
                phase[i] = phase_std * oskar_random_gaussian(0);
        }
    }
}

#ifdef __cplusplus
}
#endif
