/*
 * Copyright (c) 2012-2017, The University of Oxford
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

#include "telescope/station/oskar_evaluate_station_beam_aperture_array.h"

#include "telescope/station/oskar_evaluate_beam_horizon_direction.h"
#include "telescope/station/oskar_evaluate_element_weights.h"
#include "telescope/station/element/oskar_element_evaluate.h"
#include "telescope/station/oskar_blank_below_horizon.h"
#include "telescope/station/private_station_work.h"

#include "math/oskar_cmath.h"
#include "math/oskar_dftw.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_CHUNK_SIZE 49152

/* Private function, used for recursive calls. */
static void oskar_evaluate_station_beam_aperture_array_private(oskar_Mem* beam,
        const oskar_Station* s, int num_points, const oskar_Mem* x,
        const oskar_Mem* y, const oskar_Mem* z, double gast,
        double frequency_hz, oskar_StationWork* work, int time_index,
        int depth, int* status);


void oskar_evaluate_station_beam_aperture_array(oskar_Mem* beam,
        const oskar_Station* station, int num_points, const oskar_Mem* x,
        const oskar_Mem* y, const oskar_Mem* z, double gast,
        double frequency_hz, oskar_StationWork* work, int time_index,
        int* status)
{
    int start;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Evaluate beam immediately, without chunking, if there are no
     * child stations. */
    if (!oskar_station_has_child(station))
    {
        oskar_evaluate_station_beam_aperture_array_private(beam, station,
                num_points, x, y, z, gast, frequency_hz, work,
                time_index, 0, status);
    }
    else
    {
        oskar_Mem *c_beam, *c_x, *c_y, *c_z;
        c_beam = oskar_mem_create_alias(0, 0, 0, status);
        c_x = oskar_mem_create_alias(0, 0, 0, status);
        c_y = oskar_mem_create_alias(0, 0, 0, status);
        c_z = oskar_mem_create_alias(0, 0, 0, status);

        /* Split up list of input points into manageable chunks. */
        for (start = 0; start < num_points; start += MAX_CHUNK_SIZE)
        {
            int chunk_size;

            /* Get size of current chunk. */
            chunk_size = num_points - start;
            if (chunk_size > MAX_CHUNK_SIZE) chunk_size = MAX_CHUNK_SIZE;

            /* Get pointers to start of chunk input data. */
            oskar_mem_set_alias(c_beam, beam, start, chunk_size, status);
            oskar_mem_set_alias(c_x, x, start, chunk_size, status);
            oskar_mem_set_alias(c_y, y, start, chunk_size, status);
            oskar_mem_set_alias(c_z, z, start, chunk_size, status);

            /* Start recursive call at depth 1 (depth 0 is element level). */
            oskar_evaluate_station_beam_aperture_array_private(c_beam, station,
                    chunk_size, c_x, c_y, c_z, gast, frequency_hz, work,
                    time_index, 1, status);
        }

        /* Release handles for chunk memory. */
        oskar_mem_free(c_beam, status);
        oskar_mem_free(c_x, status);
        oskar_mem_free(c_y, status);
        oskar_mem_free(c_z, status);
    }
}

static void oskar_evaluate_station_beam_aperture_array_private(oskar_Mem* beam,
        const oskar_Station* s, int num_points, const oskar_Mem* x,
        const oskar_Mem* y, const oskar_Mem* z, double gast,
        double frequency_hz, oskar_StationWork* work, int time_index,
        int depth, int* status)
{
    double beam_x, beam_y, beam_z, wavenumber;
    oskar_Mem *weights, *weights_error, *theta, *phi, *array;
    int num_elements, is_3d;

    num_elements  = oskar_station_num_elements(s);
    is_3d         = oskar_station_array_is_3d(s);
    weights       = work->weights;
    weights_error = work->weights_error;
    theta         = work->theta_modified;
    phi           = work->phi_modified;
    array         = work->array_pattern;
    wavenumber    = 2.0 * M_PI * frequency_hz / 299792458.0;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Compute direction cosines for the beam for this station. */
    oskar_evaluate_beam_horizon_direction(&beam_x, &beam_y, &beam_z, s,
            gast, status);

    /* Evaluate beam if there are no child stations. */
    if (!oskar_station_has_child(s))
    {
        /* First optimisation: A single element model type, and either a common
         * orientation for all elements within the station, or isotropic
         * elements. */
        /* Array pattern and element pattern are separable. */
        if (oskar_station_num_element_types(s) == 1 &&
                (oskar_station_common_element_orientation(s) ||
                        oskar_element_type(oskar_station_element_const(s, 0))
                        == OSKAR_ELEMENT_TYPE_ISOTROPIC) )
        {
            /* (Always) evaluate element pattern into the output beam array. */
            oskar_element_evaluate(oskar_station_element_const(s, 0), beam,
                    oskar_station_element_x_alpha_rad(s, 0) + M_PI/2.0, /* FIXME Will change: This matches the old convention. */
                    oskar_station_element_y_alpha_rad(s, 0),
                    num_points, x, y, z, frequency_hz, theta, phi, status);

            /* Check if array pattern is enabled. */
            if (oskar_station_enable_array_pattern(s))
            {
                /* Generate beamforming weights and evaluate array pattern. */
                oskar_evaluate_element_weights(weights, weights_error,
                        wavenumber, s, beam_x, beam_y, beam_z,
                        time_index, status);
                oskar_dftw(num_elements, wavenumber,
                        oskar_station_element_true_x_enu_metres_const(s),
                        oskar_station_element_true_y_enu_metres_const(s),
                        oskar_station_element_true_z_enu_metres_const(s),
                        weights, num_points, x, y, (is_3d ? z : 0), 0, array,
                        status);

                /* Normalise array response if required. */
                if (oskar_station_normalise_array_pattern(s))
                    oskar_mem_scale_real(array, 1.0 / num_elements, status);

                /* Element-wise multiply to join array and element pattern. */
                oskar_mem_multiply(beam, beam, array, num_points, status);
            }
        }

#if 0
        /* Second optimisation: Common orientation for all elements within the
         * station, but more than one element type. */
        else if (oskar_station_common_element_orientation(s))
        {
            /* Must evaluate array pattern, so check that this is enabled. */
            if (!oskar_station_enable_array_pattern(s))
            {
                *status = OSKAR_ERR_SETTINGS_TELESCOPE;
                return;
            }

            /* Call a DFT using indexed input. */
            /* TODO Not yet implemented. */
            *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
        }
#endif

        /* No optimisation: No common element orientation. */
        /* Can't separate array and element evaluation. */
        else
        {
            int i, num_element_types;
            oskar_Mem *element_block = 0, *element = 0;
            const int* element_type_array = 0;

            /* Must evaluate array pattern, so check that this is enabled. */
            if (!oskar_station_enable_array_pattern(s))
            {
                *status = OSKAR_ERR_SETTINGS_TELESCOPE;
                return;
            }

            /* Get sized element pattern block (at depth 0). */
            element_block = oskar_station_work_beam(work, beam,
                    num_elements * num_points, 0, status);

            /* Create alias into element block. */
            element = oskar_mem_create_alias(element_block, 0, 0, status);

            /* Loop over elements and evaluate response for each. */
            element_type_array = oskar_station_element_types_cpu_const(s);
            num_element_types = oskar_station_num_element_types(s);
            for (i = 0; i < num_elements; ++i)
            {
                int element_type_idx;
                element_type_idx = element_type_array[i];
                if (element_type_idx >= num_element_types)
                {
                    *status = OSKAR_ERR_OUT_OF_RANGE;
                    break;
                }
                oskar_mem_set_alias(element, element_block, i * num_points,
                        num_points, status);
                oskar_element_evaluate(
                        oskar_station_element_const(s, element_type_idx),
                        element,
                        oskar_station_element_x_alpha_rad(s, i) + M_PI/2.0, /* FIXME Will change: This matches the old convention. */
                        oskar_station_element_y_alpha_rad(s, i),
                        num_points, x, y, z, frequency_hz, theta, phi, status);
            }

            /* Generate beamforming weights. */
            oskar_evaluate_element_weights(weights, weights_error,
                    wavenumber, s, beam_x, beam_y, beam_z,
                    time_index, status);

            /* Use DFT to evaluate array response. */
            oskar_dftw(num_elements, wavenumber,
                    oskar_station_element_true_x_enu_metres_const(s),
                    oskar_station_element_true_y_enu_metres_const(s),
                    oskar_station_element_true_z_enu_metres_const(s),
                    weights, num_points, x, y, (is_3d ? z : 0),
                    element_block, beam, status);

            /* Free element alias. */
            oskar_mem_free(element, status);

            /* Normalise array response if required. */
            if (oskar_station_normalise_array_pattern(s))
                oskar_mem_scale_real(beam, 1.0 / num_elements, status);
        }

        /* Blank (set to zero) points below the horizon. */
        oskar_blank_below_horizon(beam, z, num_points, status);
    }

    /* If there are child stations, must first evaluate the beam for each. */
    else
    {
        int i;
        oskar_Mem* signal;

        /* Must evaluate array pattern, so check that this is enabled. */
        if (!oskar_station_enable_array_pattern(s))
        {
            *status = OSKAR_ERR_SETTINGS_TELESCOPE;
            return;
        }

        /* Get sized work array for this depth, with the correct type. */
        signal = oskar_station_work_beam(work, beam, num_elements * num_points,
                depth, status);

        /* Check if child stations are identical. */
        if (oskar_station_identical_children(s))
        {
            /* Set up the output buffer for the first station. */
            oskar_Mem* output0;
            output0 = oskar_mem_create_alias(signal, 0, num_points, status);

            /* Recursive call. */
            oskar_evaluate_station_beam_aperture_array_private(output0,
                    oskar_station_child_const(s, 0), num_points,
                    x, y, z, gast, frequency_hz, work, time_index,
                    depth + 1, status);

            /* Copy beam for child station 0 into memory for other stations. */
            for (i = 1; i < num_elements; ++i)
            {
                oskar_mem_copy_contents(signal, output0, i * num_points, 0,
                        oskar_mem_length(output0), status);
            }
            oskar_mem_free(output0, status);
        }
        else
        {
            /* Loop over child stations. */
            for (i = 0; i < num_elements; ++i)
            {
                /* Set up the output buffer for this station. */
                oskar_Mem* output;
                output = oskar_mem_create_alias(signal, i * num_points,
                        num_points, status);

                /* Recursive call. */
                oskar_evaluate_station_beam_aperture_array_private(output,
                        oskar_station_child_const(s, i), num_points,
                        x, y, z, gast, frequency_hz, work, time_index,
                        depth + 1, status);
                oskar_mem_free(output, status);
            }
        }

        /* Generate beamforming weights and form beam from child stations. */
        oskar_evaluate_element_weights(weights, weights_error, wavenumber,
                s, beam_x, beam_y, beam_z, time_index, status);
        oskar_dftw(num_elements, wavenumber,
                oskar_station_element_true_x_enu_metres_const(s),
                oskar_station_element_true_y_enu_metres_const(s),
                oskar_station_element_true_z_enu_metres_const(s),
                weights, num_points, x, y, (is_3d ? z : 0), signal, beam,
                status);

        /* Normalise array response if required. */
        if (oskar_station_normalise_array_pattern(s))
            oskar_mem_scale_real(beam, 1.0 / num_elements, status);
    }
}

#ifdef __cplusplus
}
#endif
