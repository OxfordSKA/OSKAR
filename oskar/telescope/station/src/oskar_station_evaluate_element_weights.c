/*
 * Copyright (c) 2012-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/oskar_cmath.h"
#include "telescope/station/oskar_station_evaluate_element_weights.h"
#include "telescope/station/oskar_evaluate_element_weights_dft.h"
#include "telescope/station/oskar_evaluate_element_weights_errors.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_evaluate_element_weights(const oskar_Station* station,
        int feed, double frequency_hz, double x_beam, double y_beam,
        double z_beam, int time_index, oskar_Mem* weights,
        oskar_Mem* weights_scratch, int* status)
{
    if (*status) return;
    const int num_elements = oskar_station_num_elements(station);
    const double wavenumber = 2.0 * M_PI * frequency_hz / 299792458.0;
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

    /* Apply gain model. */
    if (oskar_gains_defined(oskar_station_gains_const(station)))
    {
        oskar_mem_ensure(weights_scratch, num_elements, status);
        oskar_mem_clear_contents(weights_scratch, status);
        oskar_gains_evaluate(oskar_station_gains_const(station),
                time_index, frequency_hz, weights_scratch, feed, status);
        oskar_mem_multiply(weights, weights, weights_scratch,
                0, 0, 0, num_elements, status);
    }

    /* Apply apodisation. */
    if (oskar_station_apply_element_weight(station))
    {
        oskar_mem_multiply(weights, weights,
                oskar_station_element_weight_const(station, feed),
                0, 0, 0, num_elements, status);
    }
}

#ifdef __cplusplus
}
#endif
