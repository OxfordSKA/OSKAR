/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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
    int dim = 0, num_dim = 2;
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
        oskar_Mem *ptr_meas = 0, *ptr_true = 0;
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
            {
                *status = OSKAR_ERR_BAD_DATA_TYPE;
            }
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
