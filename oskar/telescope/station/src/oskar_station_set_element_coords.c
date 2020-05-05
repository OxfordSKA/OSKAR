/*
 * Copyright (c) 2011-2020, The University of Oxford
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

#include <stdlib.h>

#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_set_element_coords(oskar_Station* station, int feed,
        int index, const double measured_enu[3], const double true_enu[3],
        int* status)
{
    int dim, num_dim = 2;
    if (*status || !station) return;

    /* Check range. */
    if (index >= station->num_elements || feed > 1)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return;
    }

    /* Check if any z component is nonzero, and set 3D flag if so. */
    if (measured_enu[2] != 0.0 || true_enu[2] != 0.0)
    {
        station->array_is_3d = OSKAR_TRUE;
        num_dim = 3;
    }

    const int type = station->precision;
    const int loc = station->mem_location;
    for (dim = 0; dim < num_dim; dim++)
    {
        oskar_Mem *ptr_meas, *ptr_true;
        ptr_meas = station->element_measured_enu_metres[feed][dim];
        ptr_true = station->element_true_enu_metres[feed][dim];
        if (!ptr_meas)
        {
            station->element_measured_enu_metres[feed][dim] =
                    oskar_mem_create(type, loc, station->num_elements, status);
            ptr_meas = station->element_measured_enu_metres[feed][dim];
        }
        if (!ptr_true)
        {
            station->element_true_enu_metres[feed][dim] =
                    oskar_mem_create(type, loc, station->num_elements, status);
            ptr_true = station->element_true_enu_metres[feed][dim];
        }
        if (loc == OSKAR_CPU)
        {
            if (type == OSKAR_DOUBLE)
            {
                ((double*)oskar_mem_void(ptr_meas))[index] = measured_enu[dim];
                ((double*)oskar_mem_void(ptr_true))[index] = true_enu[dim];
            }
            else if (type == OSKAR_SINGLE)
            {
                const float meas_enu_f = (float)(measured_enu[dim]);
                const float true_enu_f = (float)(true_enu[dim]);
                ((float*)oskar_mem_void(ptr_meas))[index] = meas_enu_f;
                ((float*)oskar_mem_void(ptr_true))[index] = true_enu_f;
            }
            else
                *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
        else
        {
            oskar_mem_set_element_real(ptr_meas,
                    index, measured_enu[dim], status);
            oskar_mem_set_element_real(ptr_true,
                    index, true_enu[dim], status);
        }
    }
}

#ifdef __cplusplus
}
#endif
