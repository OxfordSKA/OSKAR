/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#include <oskar_evaluate_station_beam_aperture_array.h>

#include <oskar_evaluate_beam_horizontal_lmn.h>
#include <oskar_evaluate_array_pattern.h>
#include <oskar_evaluate_array_pattern_hierarchical.h>
#include <oskar_evaluate_array_pattern_dipoles.h>
#include <oskar_evaluate_element_weights.h>
#include <oskar_element_evaluate.h>
#include <oskar_blank_below_horizon.h>
#include <private_station_work.h>

#include <oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_CHUNK_SIZE 49152

/* Private function, used for recursive calls. */
static void oskar_evaluate_station_beam_aperture_array_private(oskar_Mem* beam,
        const oskar_Station* station, int num_points, const oskar_Mem* x,
        const oskar_Mem* y, const oskar_Mem* z, double gast,
        oskar_StationWork* work, oskar_RandomState* random_states,
        int depth, int* status);


void oskar_evaluate_station_beam_aperture_array(oskar_Mem* beam,
        const oskar_Station* station, int num_points, const oskar_Mem* x,
        const oskar_Mem* y, const oskar_Mem* z, double gast,
        oskar_StationWork* work, oskar_RandomState* random_states,
        int* status)
{
    int start;

    /* Check all inputs. */
    if (!beam || !station || !x || !y || !z || !work || !random_states ||
            !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Evaluate beam immediately, without chunking, if there are no
     * child stations. */
    if (!oskar_station_has_child(station))
    {
        oskar_evaluate_station_beam_aperture_array_private(beam, station,
                num_points, x, y, z, gast, work, random_states, 0, status);
    }
    else
    {
        /* Split up list of input points into manageable chunks. */
        for (start = 0; start < num_points; start += MAX_CHUNK_SIZE)
        {
            int chunk_size;
            oskar_Mem c_beam, c_x, c_y, c_z;

            /* Get size of current chunk. */
            chunk_size = num_points - start;
            if (chunk_size > MAX_CHUNK_SIZE) chunk_size = MAX_CHUNK_SIZE;

            /* Get pointers to start of chunk input data. */
            oskar_mem_get_pointer(&c_beam, beam, start, chunk_size, status);
            oskar_mem_get_pointer(&c_x, x, start, chunk_size, status);
            oskar_mem_get_pointer(&c_y, y, start, chunk_size, status);
            oskar_mem_get_pointer(&c_z, z, start, chunk_size, status);

            /* Start recursive call at depth 0. */
            oskar_evaluate_station_beam_aperture_array_private(&c_beam, station,
                    chunk_size, &c_x, &c_y, &c_z, gast, work, random_states, 0,
                    status);
        }
    }
}

static void oskar_evaluate_station_beam_aperture_array_private(oskar_Mem* beam,
        const oskar_Station* station, int num_points, const oskar_Mem* x,
        const oskar_Mem* y, const oskar_Mem* z, double gast,
        oskar_StationWork* work, oskar_RandomState* random_states,
        int depth, int* status)
{
    double beam_x, beam_y, beam_z;
    oskar_Mem *weights, *weights_error;
    int num_elements;

    num_elements = oskar_station_num_elements(station);
    weights = &work->weights;
    weights_error = &work->weights_error;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check the coordinate units. */
    if (oskar_station_element_coord_units(station) != OSKAR_RADIANS)
    {
        *status = OSKAR_ERR_BAD_UNITS;
        return;
    }

    /* Check that the maximum depth in the hierarchy has not been exceeded. */
    if (depth >= OSKAR_MAX_STATION_DEPTH)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return;
    }

    /* Compute direction cosines for the beam for this station. */
    oskar_evaluate_beam_horizontal_lmn(&beam_x, &beam_y, &beam_z, station,
            gast, status);

    /* Check if there are no child stations. */
    if (!oskar_station_has_child(station))
    {
        /* Common element model for all detectors in the station. */
        if (oskar_station_single_element_model(station))
        {
            /* Array pattern and element pattern are separable. */
            oskar_Mem *array = 0, *element = 0;

            /* Evaluate array pattern if required. */
            if (oskar_station_enable_array_pattern(station))
            {
                /* Get pointer to array pattern. */
                array = (oskar_mem_is_scalar(beam) ?
                        beam : &work->array_pattern);

                /* Generate beamforming weights and evaluate array pattern. */
                oskar_evaluate_element_weights(weights, weights_error, station,
                        beam_x, beam_y, beam_z, random_states, status);
                oskar_evaluate_array_pattern(array, station, num_points,
                        x, y, z, weights, status);

                /* Normalise array response if required. */
                if (oskar_station_normalise_beam(station))
                    oskar_mem_scale_real(array, 1.0 / num_elements, status);
            }

            /* Get pointer to element pattern. */
            if (oskar_station_use_polarised_elements(station))
                element = (oskar_mem_is_matrix(beam) ?
                        beam : &work->element_pattern_matrix);
            else
                element = (!array ? beam : &work->element_pattern_scalar);

            /* Evaluate element pattern. */
            oskar_element_evaluate(oskar_station_element_const(station, 0),
                    element, oskar_station_element_orientation_x_rad(station),
                    oskar_station_element_orientation_y_rad(station),
                    num_points, x, y, z, &work->theta_modified,
                    &work->phi_modified, status);

            /* Element-wise multiply to join array and element pattern. */
            if (array && element)
            {
                oskar_mem_element_multiply(beam, array, element, num_points,
                        status);
            }
            else if (array && oskar_mem_is_matrix(beam))
            {
                /* Join array pattern with an identity matrix in output beam. */
                oskar_mem_set_value_real(beam, 1.0, status);
                oskar_mem_element_multiply(0, beam, array, num_points, status);
            }
        }

        /* Different element model per detector in the station. */
        /* Can't separate array and element evaluation. */
        else
        {
            const oskar_Element* element;
            element = oskar_station_element_const(station, 0);

            /* Must evaluate array pattern, so check that this is enabled. */
            if (!oskar_station_enable_array_pattern(station))
                *status = OSKAR_ERR_SETTINGS;

            /* Can't use tapered elements here, for the moment. */
            if (oskar_element_taper_type(element) != OSKAR_ELEMENT_TAPER_NONE)
                *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;

            /* Check if safe to proceed. */
            if (*status) return;

            /* Check that there are no spline coefficients. */
            if (!oskar_element_has_spline_data(element))
            {
                /* Generate beamforming weights and evaluate beam from
                 * dipoles that are oriented differently. */
                oskar_evaluate_element_weights(weights, weights_error, station,
                        beam_x, beam_y, beam_z, random_states, status);
                oskar_evaluate_array_pattern_dipoles(beam, station, num_points,
                        x, y, z, weights, status);

                /* Normalise array response if required. */
                if (oskar_station_normalise_beam(station))
                    oskar_mem_scale_real(beam, 1.0 / num_elements, status);
            }
            else
            {
                /* Unique spline patterns: not implemented. */
                *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
            }
        }

        /* Blank (set to zero) points below the horizon. */
        oskar_blank_below_horizon(beam, z, num_points, status);
    }

    /* If there are child stations, must first evaluate the beam for each. */
    else
    {
        int i, work_size;
        oskar_Mem* signal;

        /* Get pointer to a work array of the right type. */
        signal = (oskar_mem_is_matrix(beam)) ?
                &work->hierarchy_work_matrix[depth] :
                &work->hierarchy_work_scalar[depth];

        /* Ensure enough space in the work array. */
        work_size = num_elements * num_points;
        if ((int)oskar_mem_length(signal) < work_size)
            oskar_mem_realloc(signal, work_size, status);

        /* Loop over child stations. */
        for (i = 0; i < num_elements; ++i)
        {
            /* Set up the output buffer for this station. */
            oskar_Mem output;
            oskar_mem_get_pointer(&output, signal, i * num_points,
                    num_points, status);

            /* Recursive call. */
            oskar_evaluate_station_beam_aperture_array_private(&output,
                    oskar_station_child_const(station, i), num_points, x, y, z,
                    gast, work, random_states, depth + 1, status);
        }

        /* Generate beamforming weights and form beam from child stations. */
        oskar_evaluate_element_weights(weights, weights_error, station,
                beam_x, beam_y, beam_z, random_states, status);
        oskar_evaluate_array_pattern_hierarchical(beam, station, num_points,
                x, y, z, signal, weights, status);

        /* Normalise array response if required. */
        if (oskar_station_normalise_beam(station))
            oskar_mem_scale_real(beam, 1.0 / num_elements, status);
    }
}

#ifdef __cplusplus
}
#endif
