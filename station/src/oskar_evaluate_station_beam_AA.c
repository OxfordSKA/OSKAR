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


#include "station/oskar_evaluate_station_beam_AA.h"

#include "station/oskar_evaluate_station_beam_scalar.h"
#include "station/oskar_element_model_evaluate.h"
#include "station/oskar_evaluate_station_beam_dipoles.h"
#include "station/oskar_blank_below_horizon.h"

#include "utility/oskar_mem_type_check.h"
#include "utility/oskar_mem_scale_real.h"
#include "utility/oskar_mem_element_multiply.h"
#include "utility/oskar_mem_set_value_real.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_evaluate_station_beam_AA(oskar_Mem* beam,
        const oskar_StationModel* station, double beam_x, double beam_y,
        double beam_z, int num_points, const oskar_Mem* x, const oskar_Mem* y,
        const oskar_Mem* z, oskar_WorkStationBeam* work,
        oskar_Device_curand_state* curand_states, int* status)
{
    /* Common element model for all detectors in the station. */
    if (station->single_element_model)
    {
        /* E and G are separable. */
        oskar_Mem *E_ptr = NULL, *G_ptr = NULL;

        /* Evaluate E if required. */
        if (station->evaluate_array_factor)
        {
            /* Get pointer to E. */
            if (oskar_mem_is_scalar(beam->type))
                E_ptr = beam; /* Use memory passed to the function. */
            else
                E_ptr = &work->E; /* Use work buffer. */

            /* Evaluate array factor. */
            *status = oskar_evaluate_station_beam_scalar(E_ptr, station,
                    beam_x, beam_y, beam_z, num_points, x, y, z,
                    &work->weights, &work->weights_error, curand_states);
            if (*status) return;

            /* Normalise array beam if required. */
            if (station->normalise_beam)
            {
                oskar_mem_scale_real(E_ptr, 1.0/station->num_elements, status);
                if (*status) return;
            }
        }

        /* Evaluate G if required. */
        if (station->evaluate_element_factor && station->use_polarised_elements)
        {
            /* Get pointer to G. */
            if (oskar_mem_is_matrix(beam->type))
                G_ptr = beam; /* Use memory passed to the function. */
            else
                G_ptr = &work->G; /* Use work buffer. */

            /* Evaluate element factor. */
            *status = oskar_element_model_evaluate(station->element_pattern,
                    G_ptr, station->use_polarised_elements,
                    station->orientation_x, station->orientation_y,
                    num_points, x, y, z, &work->theta_modified,
                    &work->phi_modified);
            if (*status) return;
        }

        /* Element-wise multiply to join E and G. */
        if (E_ptr && G_ptr)
        {
            /* Use E_ptr and G_ptr. */
            oskar_mem_element_multiply(beam, E_ptr, G_ptr, num_points, status);
            if (*status) return;
        }
        else if (E_ptr && oskar_mem_is_matrix(beam->type))
        {
            /* Join E with an identity matrix in EG. */
            *status = oskar_mem_set_value_real(beam, 1.0);
            oskar_mem_element_multiply(NULL, beam, E_ptr, num_points, status);
            if (*status) return;
        }
        else if (!E_ptr && !G_ptr)
        {
            /* No evaluation: set EG to identity matrix. */
            *status = oskar_mem_set_value_real(beam, 1.0);
            if (*status) return;
        }
    }


    /* Different element model per detector in the station */
    /* Can't separate E and G evaluation */
    else /* (!station->single_element_model) */
    {
        /* With unique detector elements: E and G are not separable. */
        if (!(station->evaluate_array_factor && station->evaluate_element_factor))
        {
            *status = OSKAR_ERR_SETTINGS;
            return;
        }

        if (!station->use_polarised_elements)
        {
            *status = OSKAR_ERR_SETTINGS;
            return;
        }

        /* FIXME logic here is a bit messy... */
        if (!station->element_pattern->theta_re_x.coeff.data)
        {
            /* Call function to evaluate beam from dipoles that are
             * oriented differently. */
            *status = oskar_evaluate_station_beam_dipoles(beam, station,
                    beam_x, beam_y, beam_z, num_points, x, y, z,
                    &work->weights, &work->weights_error, curand_states);
            if (*status) return;

            /* Normalise array beam if required. */
            if (station->normalise_beam)
            {
                oskar_mem_scale_real(beam, 1.0/station->num_elements, status);
                if (*status) return;
            }
        }
        else
        {
            /* Unique spline patterns: not implemented. */
            *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
            return;
        }
    }

    /* Blank (zero) sources below the horizon. */
    *status = oskar_blank_below_horizon(beam, z, num_points);
    if (*status) return;
}


#ifdef __cplusplus
}
#endif
