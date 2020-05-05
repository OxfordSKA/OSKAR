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
    oskar_Mem *ptr_gain, *ptr_gain_error, *ptr_phase_offset, *ptr_phase_error;
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
        void *gain_, *gain_err_, *phase_, *phase_err_;

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
            *status = OSKAR_ERR_BAD_DATA_TYPE;
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
