/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_analyse(oskar_Station* station,
        int* finished_identical_station_check, int* status)
{
    int i = 0, num_feeds_to_check = 1, feed = 0;
    if (*status || !station) return;
    const int type = oskar_station_precision(station);
    const int num_elements = station->num_elements;

    /* Check station model is in CPU-accessible memory. */
    if (oskar_station_mem_location(station) != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    /* Set default station flags. */
    station->array_is_3d = 0;
    station->apply_element_errors = 0;
    station->apply_element_weight = 0;
    station->common_element_orientation = 1;
    station->common_pol_beams = 1;

    /* Check orientations in both polarisations. */
    for (feed = 0; feed < 2; feed++)
    {
        const double *alpha = 0, *beta = 0, *gamma = 0;
        char* mount_type = 0;

        /* Analyse orientations separately (always double precision). */
        mount_type = oskar_mem_char(station->element_mount_types_cpu);
        alpha = oskar_mem_double_const(
                oskar_station_element_euler_rad_const(station, feed, 0), status);
        beta  = oskar_mem_double_const(
                oskar_station_element_euler_rad_const(station, feed, 1), status);
        gamma = oskar_mem_double_const(
                oskar_station_element_euler_rad_const(station, feed, 2), status);
        for (i = 1; i < num_elements; ++i)
        {
            if (mount_type[i] != mount_type[0] ||
                    alpha[i] != alpha[0] ||
                    beta[i] != beta[0] ||
                    gamma[i] != gamma[0])
            {
                station->common_element_orientation = 0;
                break;
            }
        }
    }

    /* Check presence of any second polarisation data used for beamforming. */
    if (station->element_cable_length_error[1])
    {
        station->common_pol_beams = 0;
    }
    if (station->element_true_enu_metres[1][0] ||
            station->element_measured_enu_metres[1][0] ||
            station->element_gain[1] ||
            station->element_gain_error[1] ||
            station->element_phase_offset_rad[1] ||
            station->element_phase_error_rad[1] ||
            station->element_weight[1])
    {
        station->common_pol_beams = 0;
        num_feeds_to_check = 2;
    }
    for (feed = 0; feed < num_feeds_to_check; feed++)
    {
        if (type == OSKAR_DOUBLE)
        {
            double *z_true = 0, *z_meas = 0;
            double *amp = 0, *amp_err = 0, *phase = 0, *phase_err = 0;
            double2 *weights = 0;
            z_true    = (double*) oskar_mem_void(
                    oskar_station_element_true_enu_metres(station, feed, 2));
            z_meas    = (double*) oskar_mem_void(
                    oskar_station_element_measured_enu_metres(station, feed, 2));
            amp       = (double*) oskar_mem_void(
                    oskar_station_element_gain(station, feed));
            amp_err   = (double*) oskar_mem_void(
                    oskar_station_element_gain_error(station, feed));
            phase     = (double*) oskar_mem_void(
                    oskar_station_element_phase_offset_rad(station, feed));
            phase_err = (double*) oskar_mem_void(
                    oskar_station_element_phase_error_rad(station, feed));
            weights   = (double2*) oskar_mem_void(
                    oskar_station_element_weight(station, feed));

            for (i = 0; i < num_elements; ++i)
            {
                if (z_true[i] != 0.0 || z_meas[i] != 0.0)
                {
                    station->array_is_3d = 1;
                    break;
                }
            }
            for (i = 0; i < num_elements; ++i)
            {
                if (amp[i] != 1.0 || phase[i] != 0.0)
                {
                    station->apply_element_errors = 1;
                    break;
                }
            }
            for (i = 0; i < num_elements; ++i)
            {
                if (amp_err[i] != 0.0 || phase_err[i] != 0.0)
                {
                    station->apply_element_errors = 1;
                    *finished_identical_station_check = 1;
                    break;
                }
            }
            for (i = 0; i < num_elements; ++i)
            {
                if (weights[i].x != 1.0 || weights[i].y != 0.0)
                {
                    station->apply_element_weight = 1;
                    break;
                }
            }
        }
        else if (type == OSKAR_SINGLE)
        {
            float *z_true = 0, *z_meas = 0;
            float *amp = 0, *amp_err = 0, *phase = 0, *phase_err = 0;
            float2 *weights = 0;
            z_true    = (float*) oskar_mem_void(
                    oskar_station_element_true_enu_metres(station, feed, 2));
            z_meas    = (float*) oskar_mem_void(
                    oskar_station_element_measured_enu_metres(station, feed, 2));
            amp       = (float*) oskar_mem_void(
                    oskar_station_element_gain(station, feed));
            amp_err   = (float*) oskar_mem_void(
                    oskar_station_element_gain_error(station, feed));
            phase     = (float*) oskar_mem_void(
                    oskar_station_element_phase_offset_rad(station, feed));
            phase_err = (float*) oskar_mem_void(
                    oskar_station_element_phase_error_rad(station, feed));
            weights   = (float2*) oskar_mem_void(
                    oskar_station_element_weight(station, feed));

            for (i = 0; i < num_elements; ++i)
            {
                if (z_true[i] != 0.0 || z_meas[i] != 0.0)
                {
                    station->array_is_3d = 1;
                    break;
                }
            }
            for (i = 0; i < num_elements; ++i)
            {
                if (amp[i] != 1.0 || phase[i] != 0.0)
                {
                    station->apply_element_errors = 1;
                    break;
                }
            }
            for (i = 0; i < num_elements; ++i)
            {
                if (amp_err[i] != 0.0 || phase_err[i] != 0.0)
                {
                    station->apply_element_errors = 1;
                    *finished_identical_station_check = 1;
                    break;
                }
            }
            for (i = 0; i < num_elements; ++i)
            {
                if (weights[i].x != 1.0 || weights[i].y != 0.0)
                {
                    station->apply_element_weight = 1;
                    break;
                }
            }
        }
    }

    /* Check if station has child stations. */
    if (oskar_station_has_child(station))
    {
        /* Recursively analyse all child stations. */
        for (i = 0; i < num_elements; ++i)
        {
            oskar_station_analyse(oskar_station_child(station, i),
                    finished_identical_station_check, status);
        }

        /* Check if we need to examine every station. */
        if (*finished_identical_station_check)
        {
            station->identical_children = 0;
        }
        else
        {
            /* Check if child stations are identical. */
            station->identical_children = 1;
            for (i = 1; i < num_elements; ++i)
            {
                if (oskar_station_different(
                        oskar_station_child_const(station, 0),
                        oskar_station_child_const(station, i), status))
                {
                    station->identical_children = 0;
                }
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
