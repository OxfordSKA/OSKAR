/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdlib.h>
#include "math/oskar_cmath.h"

#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_set_element_errors(oskar_Station* station, int feed,
        int index, double gain, double gain_error, double phase_offset_deg,
        double phase_error_deg, int* status)
{
    oskar_Mem *ptr_gain = 0, *ptr_gain_error = 0;
    oskar_Mem *ptr_phase_offset = 0, *ptr_phase_error = 0;
    if (*status || !station) return;

    /* Convert phases to radians */
    const double phase_offset_rad = phase_offset_deg * M_PI / 180.0;
    const double phase_error_rad = phase_error_deg * M_PI / 180.0;

    /* Check range. */
    const int loc = station->mem_location;
    const int type = station->precision;
    const int num = station->num_elements;
    if (index >= station->num_elements || feed > 1)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return;
    }
    ptr_gain = station->element_gain[feed];
    ptr_gain_error = station->element_gain_error[feed];
    ptr_phase_offset = station->element_phase_offset_rad[feed];
    ptr_phase_error = station->element_phase_error_rad[feed];
    if (!ptr_gain)
    {
        station->element_gain[feed] = oskar_mem_create(
                type, loc, num, status);
        ptr_gain = station->element_gain[feed];
    }
    if (!ptr_gain_error)
    {
        station->element_gain_error[feed] = oskar_mem_create(
                type, loc, num, status);
        ptr_gain_error = station->element_gain_error[feed];
    }
    if (!ptr_phase_offset)
    {
        station->element_phase_offset_rad[feed] = oskar_mem_create(
                type, loc, num, status);
        ptr_phase_offset = station->element_phase_offset_rad[feed];
    }
    if (!ptr_phase_error)
    {
        station->element_phase_error_rad[feed] = oskar_mem_create(
                type, loc, num, status);
        ptr_phase_error = station->element_phase_error_rad[feed];
    }
    if (loc == OSKAR_CPU)
    {
        void *gain_ = 0, *gain_err_ = 0, *phase_ = 0, *phase_err_ = 0;

        /* Get raw pointers. */
        gain_      = oskar_mem_void(ptr_gain);
        gain_err_  = oskar_mem_void(ptr_gain_error);
        phase_     = oskar_mem_void(ptr_phase_offset);
        phase_err_ = oskar_mem_void(ptr_phase_error);
        if (type == OSKAR_DOUBLE)
        {
            ((double*)gain_)[index] = gain;
            ((double*)gain_err_)[index] = gain_error;
            ((double*)phase_)[index] = phase_offset_rad;
            ((double*)phase_err_)[index] = phase_error_rad;
        }
        else if (type == OSKAR_SINGLE)
        {
            ((float*)gain_)[index] = (float) gain;
            ((float*)gain_err_)[index] = (float) gain_error;
            ((float*)phase_)[index] = (float) phase_offset_rad;
            ((float*)phase_err_)[index] = (float) phase_error_rad;
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
    else
    {
        oskar_mem_set_element_real(
                ptr_gain, index, gain, status);
        oskar_mem_set_element_real(
                ptr_gain_error, index, gain_error, status);
        oskar_mem_set_element_real(
                ptr_phase_offset, index, phase_offset_rad, status);
        oskar_mem_set_element_real(
                ptr_phase_error, index, phase_error_rad, status);
    }
}

#ifdef __cplusplus
}
#endif
