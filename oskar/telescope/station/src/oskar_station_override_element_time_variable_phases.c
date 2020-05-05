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

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_override_element_time_variable_phases(
        oskar_Station* station, int feed, double phase_std_rad, int* status)
{
    int i;
    if (*status || !station) return;

    /* Override element data only at last level. */
    if (oskar_station_has_child(station))
        for (i = 0; i < station->num_elements; ++i)
            oskar_station_override_element_time_variable_phases(
                    oskar_station_child(station, i),
                    feed, phase_std_rad, status);
    else
    {
        oskar_Mem* ptr = station->element_phase_error_rad[feed];
        if (!ptr)
        {
            station->element_phase_error_rad[feed] = oskar_mem_create(
                    station->precision, station->mem_location,
                    station->num_elements, status);
            ptr = station->element_phase_error_rad[feed];
        }
        oskar_mem_set_value_real(ptr,
                phase_std_rad, 0, station->num_elements, status);
    }
}

#ifdef __cplusplus
}
#endif
