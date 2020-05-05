/*
 * Copyright (c) 2012-2020, The University of Oxford
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

#include "telescope/station/oskar_station_evaluate_element_weights.h"
#include "telescope/station/oskar_evaluate_element_weights_dft.h"
#include "telescope/station/oskar_evaluate_element_weights_errors.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_evaluate_element_weights(const oskar_Station* station,
        int feed, double wavenumber, double x_beam, double y_beam,
        double z_beam, int time_index, oskar_Mem* weights,
        oskar_Mem* weights_scratch, int* status)
{
    if (*status) return;
    const int num_elements = oskar_station_num_elements(station);
    oskar_mem_ensure(weights, num_elements, status);

    /* Generate DFT weights. */
    oskar_evaluate_element_weights_dft(num_elements,
            oskar_station_element_measured_enu_metres_const(station, feed, 0),
            oskar_station_element_measured_enu_metres_const(station, feed, 1),
            oskar_station_element_measured_enu_metres_const(station, feed, 2),
            oskar_station_element_cable_length_error_metres_const(station, feed),
            wavenumber, x_beam, y_beam, z_beam, weights, status);

    /* Apply time-variable errors. */
    if (oskar_station_apply_element_errors(station))
    {
        oskar_mem_ensure(weights_scratch, num_elements, status);
        oskar_evaluate_element_weights_errors(num_elements,
                oskar_station_element_gain_const(station, feed),
                oskar_station_element_gain_error_const(station, feed),
                oskar_station_element_phase_offset_rad_const(station, feed),
                oskar_station_element_phase_error_rad_const(station, feed),
                oskar_station_seed_time_variable_errors(station), time_index,
                oskar_station_unique_id(station), weights_scratch, status);
        oskar_mem_multiply(weights, weights, weights_scratch,
                0, 0, 0, num_elements, status);
    }

    /* Apply apodisation. */
    if (oskar_station_apply_element_weight(station))
        oskar_mem_multiply(weights, weights,
                oskar_station_element_weight_const(station, feed),
                0, 0, 0, num_elements, status);
}

#ifdef __cplusplus
}
#endif
