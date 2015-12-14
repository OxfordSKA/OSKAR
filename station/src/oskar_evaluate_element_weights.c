/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include <oskar_evaluate_element_weights.h>
#include <oskar_evaluate_element_weights_dft.h>
#include <oskar_evaluate_element_weights_errors.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_evaluate_element_weights(oskar_Mem* weights,
        oskar_Mem* weights_error, double wavenumber,
        const oskar_Station* station, double x_beam, double y_beam,
        double z_beam, int time_index, int* status)
{
    int num_elements;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Resize weights and weights error work arrays if required. */
    num_elements = oskar_station_num_elements(station);
    if ((int)oskar_mem_length(weights) < num_elements)
        oskar_mem_realloc(weights, num_elements, status);
    if ((int)oskar_mem_length(weights_error) < num_elements)
        oskar_mem_realloc(weights_error, num_elements, status);

    /* Generate DFT weights. */
    oskar_evaluate_element_weights_dft(weights, num_elements, wavenumber,
            oskar_station_element_measured_x_enu_metres_const(station),
            oskar_station_element_measured_y_enu_metres_const(station),
            oskar_station_element_measured_z_enu_metres_const(station),
            x_beam, y_beam, z_beam, status);

    /* Apply time-variable errors. */
    if (oskar_station_apply_element_errors(station))
    {
        /* Generate weights errors. */
        oskar_evaluate_element_weights_errors(num_elements,
                oskar_station_element_gain_const(station),
                oskar_station_element_gain_error_const(station),
                oskar_station_element_phase_offset_rad_const(station),
                oskar_station_element_phase_error_rad_const(station),
                oskar_station_seed_time_variable_errors(station), time_index,
                oskar_station_unique_id(station), weights_error, status);

        /* Modify the weights (complex multiply with error vector). */
        oskar_mem_element_multiply(0, weights, weights_error,
                num_elements, status);
    }

    /* Modify the weights using the provided apodisation values. */
    if (oskar_station_apply_element_weight(station))
    {
        oskar_mem_element_multiply(0, weights,
                oskar_station_element_weight_const(station), num_elements,
                status);
    }
}

#ifdef __cplusplus
}
#endif
