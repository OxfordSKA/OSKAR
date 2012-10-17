/*
 * Copyright (c) 2012, The University of Oxford
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

#include "station/oskar_evaluate_station_beam_aperture_array.h"

#include "station/oskar_evaluate_array_pattern.h"
#include "station/oskar_evaluate_array_pattern_dipoles.h"
#include "station/oskar_element_model_evaluate.h"
#include "station/oskar_blank_below_horizon.h"

#include "utility/oskar_mem_type_check.h"
#include "utility/oskar_mem_scale_real.h"
#include "utility/oskar_mem_element_multiply.h"
#include "utility/oskar_mem_set_value_real.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_evaluate_station_beam_aperture_array(oskar_Mem* beam,
        const oskar_StationModel* station, double beam_x, double beam_y,
        double beam_z, int num_points, const oskar_Mem* x, const oskar_Mem* y,
        const oskar_Mem* z, oskar_WorkStationBeam* work,
        oskar_CurandState* curand_states, int* status)
{
    /* Check all inputs. */
    if (!beam || !station || !x || !y || !z || !work || !curand_states ||
            !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Common element model for all detectors in the station. */
    if (station->single_element_model)
    {
        /* Array pattern and element pattern are separable. */
        oskar_Mem *array = 0, *element = 0;

        /* Evaluate array pattern if required. */
        if (station->enable_array_pattern)
        {
            /* Get pointer to array pattern. */
            array = (oskar_mem_is_scalar(beam->type) ?
                    beam : &work->array_pattern);

            /* Evaluate array pattern. */
            oskar_evaluate_array_pattern(array, station,
                    beam_x, beam_y, beam_z, num_points, x, y, z,
                    &work->weights, &work->weights_error, curand_states,
                    status);

            /* Normalise array response if required. */
            if (station->normalise_beam)
                oskar_mem_scale_real(array, 1.0/station->num_elements, status);
        }

        /* Get pointer to element pattern. */
        if (station->use_polarised_elements)
            element = (oskar_mem_is_matrix(beam->type) ?
                    beam : &work->element_pattern_matrix);
        else
            element = (!array ? beam : &work->element_pattern_scalar);

        /* Evaluate element pattern. */
        oskar_element_model_evaluate(station->element_pattern,
                element, station->orientation_x, station->orientation_y,
                num_points, x, y, z, &work->theta_modified,
                &work->phi_modified, status);

        /* Element-wise multiply to join array and element pattern. */
        if (array && element)
        {
            oskar_mem_element_multiply(beam, array, element, num_points,
                    status);
        }
        else if (array && oskar_mem_is_matrix(beam->type))
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
        /* Must evaluate array pattern, so check that this is enabled. */
        if (!station->enable_array_pattern)
            *status = OSKAR_ERR_SETTINGS;

        /* Check that there are no spline coefficients. */
        if (!station->element_pattern->theta_re_x.coeff.data)
        {
            /* Evaluate beam from dipoles that are oriented differently. */
            oskar_evaluate_array_pattern_dipoles(beam, station, beam_x,
                    beam_y, beam_z, num_points, x, y, z, &work->weights,
                    &work->weights_error, curand_states, status);

            /* Normalise array response if required. */
            if (station->normalise_beam)
                oskar_mem_scale_real(beam, 1.0/station->num_elements, status);
        }
        else
        {
            /* Unique spline patterns: not implemented. */
            *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
        }
    }

    /* Blank (zero) sources below the horizon. */
    oskar_blank_below_horizon(beam, z, num_points, status);
}


#ifdef __cplusplus
}
#endif
