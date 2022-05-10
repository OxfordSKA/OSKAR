/*
 * Copyright (c) 2012-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/oskar_evaluate_station_beam_aperture_array.h"

#include "convert/oskar_convert_enu_directions_to_theta_phi.h"
#include "convert/oskar_convert_theta_phi_to_ludwig3_components.h"
#include "telescope/station/oskar_station_evaluate_element_weights.h"
#include "telescope/station/element/oskar_element_evaluate.h"
#include "telescope/station/oskar_blank_below_horizon.h"
#include "telescope/station/private_station_work.h"

#include "math/oskar_cmath.h"
#include "math/oskar_dftw.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_CHUNK_SIZE 49152

static void oskar_evaluate_station_beam_aperture_array_private(
        const oskar_Station* s, oskar_StationWork* work, int offset_points,
        int num_points, const oskar_Mem* x, const oskar_Mem* y,
        const oskar_Mem* z, int time_index, double gast_rad,
        double frequency_hz, int depth, int offset_out, oskar_Mem* beam,
        int* status);


void oskar_evaluate_station_beam_aperture_array(
        const oskar_Station* station,
        oskar_StationWork* work,
        int num_points,
        const oskar_Mem* x,
        const oskar_Mem* y,
        const oskar_Mem* z,
        int time_index,
        double gast_rad,
        double frequency_hz,
        oskar_Mem* beam,
        int* status)
{
    if (*status) return;

    /* Evaluate beam directly if there are no child stations. */
    if (!oskar_station_has_child(station))
    {
        oskar_evaluate_station_beam_aperture_array_private(station, work,
                0, num_points, x, y, z, time_index,
                gast_rad, frequency_hz, 0, 0, beam, status);
    }
    else
    {
        /* Split up list of input points into manageable chunks. */
        int start = 0;
        for (start = 0; start < num_points; start += MAX_CHUNK_SIZE)
        {
            int chunk_size = num_points - start;
            if (chunk_size > MAX_CHUNK_SIZE) chunk_size = MAX_CHUNK_SIZE;

            /* Start recursive call at depth 1 (depth 0 is element level). */
            oskar_evaluate_station_beam_aperture_array_private(station, work,
                    start, chunk_size, x, y, z, time_index,
                    gast_rad, frequency_hz, 1, start, beam, status);
        }
    }
}

static void oskar_evaluate_station_beam_aperture_array_private(
        const oskar_Station* s, oskar_StationWork* work, int offset_points,
        int num_points, const oskar_Mem* x, const oskar_Mem* y,
        const oskar_Mem* z, int time_index, double gast_rad,
        double frequency_hz, int depth, int offset_out, oskar_Mem* beam,
        int* status)
{
    double beam_x = 0.0, beam_y = 0.0, beam_z = 0.0;
    oskar_Mem *signal = 0, *theta = 0, *phi_x = 0, *phi_y = 0;
    const oskar_Mem* element_types_ptr = 0;
    int i = 0;
    if (*status) return;

    const double wavenumber = 2.0 * M_PI * frequency_hz / 299792458.0;
    const int swap_xy       = oskar_station_swap_xy(s);
    const int is_3d         = oskar_station_array_is_3d(s);
    const int norm_array    = oskar_station_normalise_array_pattern(s);
    const int norm_element  = oskar_station_normalise_element_pattern(s);
    const int num_elements  = oskar_station_num_elements(s);
    const int num_feeds     = (oskar_station_common_pol_beams(s) ||
            !oskar_mem_is_matrix(beam)) ? 1 : 2;
    theta = work->theta_modified;
    phi_x = work->phi_x;
    phi_y = work->phi_y;

    /* Compute direction cosines for the beam for this station. */
    oskar_station_beam_horizon_direction(s, gast_rad,
            &beam_x, &beam_y, &beam_z, status);

    /* Check if HARP data exist. */
    const oskar_Harp* harp_data = oskar_station_harp_data_const(
            s, frequency_hz);
    if (harp_data)
    {
        oskar_mem_ensure(theta, num_points, status);
        oskar_mem_ensure(phi_x, num_points, status);
        oskar_mem_ensure(phi_y, num_points, status);
        oskar_convert_enu_directions_to_theta_phi(
                offset_points, num_points, x, y, z, 0,
                M_PI/2.0 - (oskar_station_element_euler_index_rad(s, 0, 0, 0) + M_PI/2.0),
                M_PI/2.0 - (oskar_station_element_euler_index_rad(s, 1, 0, 0)),
                theta, phi_x, phi_y, status);
        oskar_harp_evaluate_smodes(
                harp_data,
                num_points,
                theta,
                phi_x,
                work->poly,
                work->ee,
                work->qq,
                work->dd,
                work->pth,
                work->pph,
                status);
        for (i = 0; i < 2; ++i)
        {
            oskar_station_evaluate_element_weights(s, i, frequency_hz,
                    beam_x, beam_y, beam_z, time_index,
                    work->weights, work->weights_scratch, status);
            oskar_harp_evaluate_station_beam(
                    harp_data,
                    num_points,
                    theta,
                    phi_x,
                    frequency_hz,
                    i,
                    num_elements,
                    oskar_station_element_true_enu_metres_const(s, i, 0),
                    oskar_station_element_true_enu_metres_const(s, i, 1),
                    oskar_station_element_true_enu_metres_const(s, i, 2),
                    work->weights,
                    work->pth,
                    work->pph,
                    work->phase_fac,
                    work->beam_coeffs,
                    offset_out,
                    beam,
                    status);
        }
        oskar_convert_theta_phi_to_ludwig3_components(num_points,
                phi_x, phi_y, swap_xy, offset_out, beam, status);
        oskar_blank_below_horizon(offset_points, num_points, z,
                offset_out, beam, status);
        return;
    }

    /* Evaluate beam if there are no child stations. */
    if (!oskar_station_has_child(s))
    {
        /* Check if element types can be used to evaluate the beam. */
        const int num_element_types = oskar_station_num_element_types(s);
        if (oskar_station_common_element_orientation(s))
        {
            /* Evaluate element patterns for each element type. */
            element_types_ptr = oskar_station_element_types_const(s);
            signal = oskar_station_work_beam(work, beam,
                    num_element_types * (num_points + 1), 0, status);
            for (i = 0; i < num_element_types; ++i)
            {
                oskar_element_evaluate(
                        oskar_station_element_const(s, i),
                        norm_element, swap_xy,
                        oskar_station_element_euler_index_rad(s, 0, 0, 0) + M_PI/2.0, /* FIXME Will change: This matches the old convention. */
                        oskar_station_element_euler_index_rad(s, 1, 0, 0),
                        offset_points, num_points, x, y, z, frequency_hz,
                        theta, phi_x, phi_y, i * num_points, signal, status);
            }
        }
        else
        {
            /* Evaluate element patterns for each element. */
            const int* element_type = oskar_station_element_types_cpu_const(s);
            signal = oskar_station_work_beam(work, beam,
                    num_elements * (num_points + 1), 0, status);
            for (i = 0; i < num_elements; ++i)
            {
                if (element_type[i] >= num_element_types)
                {
                    *status = OSKAR_ERR_OUT_OF_RANGE;
                    break;
                }
                oskar_element_evaluate(
                        oskar_station_element_const(s, element_type[i]),
                        norm_element, swap_xy,
                        oskar_station_element_euler_index_rad(s, 0, 0, i) + M_PI/2.0, /* FIXME Will change: This matches the old convention. */
                        oskar_station_element_euler_index_rad(s, 1, 0, i),
                        offset_points, num_points, x, y, z, frequency_hz,
                        theta, phi_x, phi_y, i * num_points, signal, status);
            }
        }
        if (oskar_station_enable_array_pattern(s))
        {
            for (i = 0; i < num_feeds; ++i)
            {
                const int eval_x = (i == 0 || num_feeds == 1) ? 1 : 0;
                const int eval_y = (i == 1 || num_feeds == 1) ? 1 : 0;
                oskar_station_evaluate_element_weights(s, i, frequency_hz,
                        beam_x, beam_y, beam_z, time_index,
                        work->weights, work->weights_scratch, status);
                oskar_dftw(norm_array, num_elements, wavenumber, work->weights,
                        oskar_station_element_true_enu_metres_const(s, i, 0),
                        oskar_station_element_true_enu_metres_const(s, i, 1),
                        oskar_station_element_true_enu_metres_const(s, i, 2),
                        offset_points, num_points, x, y, (is_3d ? z : 0),
                        element_types_ptr, signal, eval_x, eval_y,
                        offset_out, beam, status);
            }
        }
        else
        {
            if (oskar_station_num_element_types(s) == 1)
            {
                oskar_mem_copy_contents(beam, signal,
                        offset_out, 0, num_points, status);
            }
            else
            {
                /* Can't separate array and element pattern evaluation. */
                *status = OSKAR_ERR_SETTINGS_TELESCOPE;
                return;
            }
        }
        oskar_blank_below_horizon(offset_points, num_points, z,
                offset_out, beam, status);
    }
    else /* If there are child stations, first evaluate the beam for each. */
    {
        if (!oskar_station_enable_array_pattern(s))
        {
            *status = OSKAR_ERR_SETTINGS_TELESCOPE;
            return;
        }
        signal = oskar_station_work_beam(work, beam,
                num_elements * num_points, depth, status);
        if (oskar_station_identical_children(s))
        {
            oskar_evaluate_station_beam_aperture_array_private(
                    oskar_station_child_const(s, 0), work, offset_points,
                    num_points, x, y, z, time_index, gast_rad, frequency_hz,
                    depth + 1, 0, signal, status);
            for (i = 1; i < num_elements; ++i)
            {
                oskar_mem_copy_contents(signal, signal, i * num_points, 0,
                        num_points, status);
            }
        }
        else
        {
            for (i = 0; i < num_elements; ++i)
            {
                oskar_evaluate_station_beam_aperture_array_private(
                        oskar_station_child_const(s, i), work, offset_points,
                        num_points, x, y, z, time_index, gast_rad, frequency_hz,
                        depth + 1, i * num_points, signal, status);
            }
        }
        for (i = 0; i < num_feeds; ++i)
        {
            const int eval_x = (i == 0 || num_feeds == 1) ? 1 : 0;
            const int eval_y = (i == 1 || num_feeds == 1) ? 1 : 0;
            oskar_station_evaluate_element_weights(s, i, frequency_hz,
                    beam_x, beam_y, beam_z, time_index,
                    work->weights, work->weights_scratch, status);
            oskar_dftw(norm_array, num_elements, wavenumber, work->weights,
                    oskar_station_element_true_enu_metres_const(s, i, 0),
                    oskar_station_element_true_enu_metres_const(s, i, 1),
                    oskar_station_element_true_enu_metres_const(s, i, 2),
                    offset_points, num_points, x, y, (is_3d ? z : 0), 0,
                    signal, eval_x, eval_y, offset_out, beam, status);
        }
    }
}

#ifdef __cplusplus
}
#endif
