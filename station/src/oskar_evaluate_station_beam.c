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

#include "station/oskar_blank_below_horizon.h"
#include "station/oskar_element_model_evaluate.h"
#include "station/oskar_evaluate_station_beam_dipoles.h"
#include "station/oskar_evaluate_station_beam_gaussian.h"
#include "station/oskar_evaluate_station_beam_scalar.h"
#include "station/oskar_evaluate_station_beam.h"
#include "station/oskar_station_model_location.h"
#include "station/oskar_station_model_type.h"
#include "utility/oskar_mem_get_pointer.h"
#include "utility/oskar_mem_element_multiply.h"
#include "utility/oskar_mem_realloc.h"
#include "utility/oskar_mem_scale_real.h"
#include "utility/oskar_mem_set_value_real.h"
#include "utility/oskar_mem_type_check.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Private functions */
static int parse_inputs(oskar_Mem* EG, const oskar_StationModel* station,
        int num_points, const oskar_Mem* l, const oskar_Mem* m,
        const oskar_Mem* n, oskar_WorkStationBeam* work,
        oskar_Device_curand_state* curand_states);
static int evaluate_EG_AA(oskar_Mem* EG, const oskar_StationModel* station,
        double l_beam, double m_beam, double n_beam, int num_points,
        const oskar_Mem* l, const oskar_Mem* m, const oskar_Mem* n,
        oskar_WorkStationBeam* work, oskar_Device_curand_state* curand_states);


int oskar_evaluate_station_beam(oskar_Mem* EG, const oskar_StationModel* station,
        double l_beam, double m_beam, double n_beam, int num_points,
        const oskar_Mem* l, const oskar_Mem* m, const oskar_Mem* n,
        oskar_WorkStationBeam* work, oskar_Device_curand_state* curand_states)
{
    int error = OSKAR_SUCCESS;

    /* Sanity check on inputs. */
    error = parse_inputs(EG, station, num_points, l, m, n, work, curand_states);

    /* Evaluate the station beam for specified station type */
    switch (station->station_type)
    {
        /* Aperture array station */
        case OSKAR_STATION_TYPE_AA:
        {
            error = evaluate_EG_AA(EG, station, l_beam, m_beam, n_beam,
                    num_points, l, m, n, work, curand_states);
            if (error) return error;
            break;
        }

        /* Circular Gaussian beam */
        case OSKAR_STATION_TYPE_GAUSSIAN_BEAM:
        {
            /* FIXME how to set/get this ...? should be in the station model */
            double fwhm_deg = 1.0;
            /* FIXME need phase centre relative l,m */
            oskar_evaluate_station_beam_gaussian(EG, num_points, l, m, fwhm_deg, &error);
            if (error) return error;
            /* FIXME horizon clip needed */
            break;
        }

        /* Dish */
        case OSKAR_STATION_TYPE_DISH:
        {
            return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
        }

        default:
        {
            return OSKAR_ERR_SETTINGS_INTERFEROMETER;
        }
    }

    return error;
}





/* Private functions */

static int parse_inputs(oskar_Mem* EG, const oskar_StationModel* station,
        int num_points, const oskar_Mem* l, const oskar_Mem* m,
        const oskar_Mem* n, oskar_WorkStationBeam* work,
        oskar_Device_curand_state* curand_states)
{
    /* Sanity check on inputs. */
    if (!EG || !station || !l || !m || !n || !work || !curand_states)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Check the coordinate units. */
    if (station->coord_units != OSKAR_RADIANS)
        return OSKAR_ERR_BAD_UNITS;

    /* Check that there is memory available. */
    if (!EG->data || !l->data || !m->data || !n->data)
        return OSKAR_ERR_MEMORY_NOT_ALLOCATED;

    /* Check that the relevant memory is on the GPU. */
    if (oskar_station_model_location(station) != OSKAR_LOCATION_GPU ||
            EG->location != OSKAR_LOCATION_GPU ||
            l->location != OSKAR_LOCATION_GPU ||
            m->location != OSKAR_LOCATION_GPU ||
            n->location != OSKAR_LOCATION_GPU)
    {
        return OSKAR_ERR_BAD_LOCATION;
    }

    /* Check that the array sizes are OK. */
    if (EG->num_elements < num_points || l->num_elements < num_points ||
            m->num_elements < num_points || n->num_elements < num_points)
    {
        return OSKAR_ERR_DIMENSION_MISMATCH;
    }

    /* Check the data types. */
    if (oskar_mem_is_real(EG->type) || oskar_mem_is_complex(l->type) ||
            oskar_mem_is_complex(m->type) || oskar_mem_is_complex(n->type))
    {
        return OSKAR_ERR_BAD_DATA_TYPE;
    }

    return OSKAR_SUCCESS;
}


static int evaluate_EG_AA(oskar_Mem* EG, const oskar_StationModel* station,
        double l_beam, double m_beam, double n_beam, int num_points,
        const oskar_Mem* l, const oskar_Mem* m, const oskar_Mem* n,
        oskar_WorkStationBeam* work, oskar_Device_curand_state* curand_states)
{
    int error = OSKAR_SUCCESS;

    /* Common element model for all detectors in the station. */
    if (station->single_element_model)
    {
        /* E and G are separable. */
        oskar_Mem *E_ptr = NULL, *G_ptr = NULL;

        /* Evaluate E if required. */
        if (station->evaluate_array_factor)
        {
            /* Get pointer to E. */
            if (oskar_mem_is_scalar(EG->type))
                E_ptr = EG; /* Use memory passed to the function. */
            else
                E_ptr = &work->E; /* Use work buffer. */

            /* Evaluate array factor. */
            error = oskar_evaluate_station_beam_scalar(E_ptr, station,
                    l_beam, m_beam, n_beam, num_points, l, m, n,
                    &work->weights, &work->weights_error, curand_states);
            if (error) return error;

            /* Normalise array beam if required. */
            if (station->normalise_beam)
            {
                error = oskar_mem_scale_real(E_ptr, 1.0/station->num_elements);
                if (error) return error;
            }
        }

        /* Evaluate G if required. */
        if (station->evaluate_element_factor && station->use_polarised_elements)
        {
            /* Get pointer to G. */
            if (oskar_mem_is_matrix(EG->type))
                G_ptr = EG; /* Use memory passed to the function. */
            else
                G_ptr = &work->G; /* Use work buffer. */

            /* Evaluate element factor. */
            error = oskar_element_model_evaluate(station->element_pattern,
                    G_ptr, station->use_polarised_elements,
                    station->orientation_x, station->orientation_y,
                    num_points, l, m, n, &work->theta_modified,
                    &work->phi_modified);
            if (error) return error;
        }

        /* Element-wise multiply to join E and G. */
        if (E_ptr && G_ptr)
        {
            /* Use E_ptr and G_ptr. */
            error = oskar_mem_element_multiply(EG, E_ptr, G_ptr,
                    num_points);
            if (error) return error;
        }
        else if (E_ptr && oskar_mem_is_matrix(EG->type))
        {
            /* Join E with an identity matrix in EG. */
            error = oskar_mem_set_value_real(EG, 1.0);
            if (error) return error;
            error = oskar_mem_element_multiply(NULL, EG, E_ptr, num_points);
            if (error) return error;
        }
        else if (!E_ptr && !G_ptr)
        {
            /* No evaluation: set EG to identity matrix. */
            error = oskar_mem_set_value_real(EG, 1.0);
            if (error) return error;
        }
    }


    /* Different element model per detector in the station */
    /* Can't separate E and G evaluation */
    else /* (!station->single_element_model) */
    {
        /* With unique detector elements: E and G are not separable. */
        if (!(station->evaluate_array_factor && station->evaluate_element_factor))
            return OSKAR_ERR_SETTINGS;

        if (!station->use_polarised_elements)
            return OSKAR_ERR_SETTINGS;

        /* FIXME logic here is a bit messy... */
        if (!station->element_pattern->theta_re_x.coeff.data)
        {
            /* Call function to evaluate beam from dipoles that are
             * oriented differently. */
            error = oskar_evaluate_station_beam_dipoles(EG, station,
                    l_beam, m_beam, n_beam, num_points, l, m, n,
                    &work->weights, &work->weights_error, curand_states);
            if (error) return error;

            /* Normalise array beam if required. */
            if (station->normalise_beam)
            {
                error = oskar_mem_scale_real(EG, 1.0/station->num_elements);
                if (error) return error;
            }
        }
        else
        {
            /* Unique spline patterns: not implemented. */
            return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
        }
    }

    /* Blank (zero) sources below the horizon. */
    error = oskar_blank_below_horizon(EG, n, num_points);
    if (error) return error;

    return OSKAR_SUCCESS;
}


#ifdef __cplusplus
}
#endif
