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
#include "station/oskar_evaluate_dipole_pattern.h"
#include "station/oskar_evaluate_spline_pattern.h"
#include "station/oskar_evaluate_station_beam_dipoles.h"
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

int oskar_evaluate_station_beam(oskar_Mem* EG, const oskar_StationModel* station,
        double l_beam, double m_beam, double n_beam, const oskar_Mem* l,
        const oskar_Mem* m, const oskar_Mem* n, oskar_Work* work,
        oskar_Device_curand_state* curand_states)
{
    int error = 0, num_points;

    /* Sanity check on inputs. */
    if (EG == NULL || station == NULL || l == NULL || m == NULL || n == NULL ||
            work == NULL)
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
        return OSKAR_ERR_BAD_LOCATION;

    /* Check that the array sizes match. */
    num_points = l->num_elements;
    if (EG->num_elements != num_points || m->num_elements != num_points ||
            n->num_elements != num_points)
        return OSKAR_ERR_DIMENSION_MISMATCH;

    /* Check the data types. */
    if (oskar_mem_is_real(EG->type) || oskar_mem_is_complex(l->type) ||
            oskar_mem_is_complex(m->type) || oskar_mem_is_complex(n->type))
        return OSKAR_ERR_BAD_DATA_TYPE;

    /* Check if the station is a dish. */
    if (station->station_type == OSKAR_STATION_TYPE_DISH)
    {
        /* Evaluate dish beam. */
        return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    }

    /* Check if the station is an aperture array. */
    else if (station->station_type == OSKAR_STATION_TYPE_AA)
    {
        oskar_Mem weights, weights_error;
        int workspace_complex = 0, workspace_matrix = 0;

        /* Determine the amount of workspace memory required. */
        workspace_complex += 2 * station->num_elements; /* For weights. */
        if (!station->use_polarised_elements || station->single_element_model)
        {
            if (station->evaluate_array_factor && oskar_mem_is_matrix(EG->type))
                workspace_complex += num_points; /* For array factor. */
            if (station->evaluate_element_factor &&
                    !station->use_polarised_elements)
                workspace_complex += num_points; /* For element factor. */
            if (station->evaluate_element_factor &&
                    oskar_mem_is_scalar(EG->type) &&
                    station->use_polarised_elements)
                workspace_matrix += num_points; /* For element factor. */
        }

        /* Resize the work arrays if needed. */
        if (work->complex.num_elements - work->used_complex < workspace_complex)
        {
            if (work->used_complex != 0)
                return OSKAR_ERR_MEMORY_ALLOC_FAILURE; /* Work buffer in use. */
            error = oskar_mem_realloc(&work->complex, workspace_complex);
            if (error) return error;
        }
        if (work->matrix.num_elements - work->used_matrix < workspace_matrix)
        {
            if (work->used_matrix != 0)
                return OSKAR_ERR_MEMORY_ALLOC_FAILURE; /* Work buffer in use. */
            error = oskar_mem_realloc(&work->matrix, workspace_matrix);
            if (error) return error;
        }

        /* Non-owned pointers to the weights and weights error work arrays. */
        error = oskar_mem_get_pointer(&weights, &work->complex,
                work->used_complex, station->num_elements);
        work->used_complex += station->num_elements;
        if (error) return error;
        error = oskar_mem_get_pointer(&weights_error, &work->complex,
                work->used_complex, station->num_elements);
        work->used_complex += station->num_elements;
        if (error) return error;

        /* Check whether using a common or unique element model. */
        if (!station->single_element_model && station->use_polarised_elements)
        {
            /* Unique receptor elements: E and G are not separable. */
            if (!station->evaluate_array_factor ||
                    !station->evaluate_element_factor)
                return OSKAR_ERR_SETTINGS;

            if (!station->element_pattern)
            {
                /* Call function to evaluate beam from dipoles that are
                 * oriented differently. */
                error = oskar_evaluate_station_beam_dipoles(EG, station,
                        l_beam, m_beam, n_beam, l, m, n, &weights,
                        &weights_error, curand_states);
                if (error) return error;

                /* Normalise beam if required. */
                if (station->normalise_beam)
                {
                    error = oskar_mem_scale_real(EG,
                            1.0 / station->num_elements);
                    if (error) return error;
                }
            }
            else
            {
                /* Unique element patterns: not implemented. */
                return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
            }
        }
        else
        {
            /* Common receptor elements: E and G are separable. */
            oskar_Mem E_temp, G_temp, *E_ptr = NULL, *G_ptr = NULL;

            /* Evaluate E if required. */
            if (station->evaluate_array_factor)
            {
                /* Get pointer to E. */
                if (oskar_mem_is_scalar(EG->type))
                {
                    /* Use the memory passed to the function. */
                    E_ptr = EG;
                }
                else
                {
                    /* Use work buffer. */
                    error = oskar_mem_get_pointer(&E_temp, &work->complex,
                            work->used_complex, num_points);
                    work->used_complex += num_points;
                    if (error) return error;
                    E_ptr = &E_temp;
                }

                /* Evaluate array factor. */
                error = oskar_evaluate_station_beam_scalar(E_ptr, station,
                        l_beam, m_beam, n_beam, l, m, n, &weights,
                        &weights_error, curand_states);
                if (error) return error;

                /* Normalise beam if required. */
                if (station->normalise_beam)
                {
                    error = oskar_mem_scale_real(E_ptr,
                            1.0 / station->num_elements);
                    if (error) return error;
                }
            }

            /* Evaluate G if required. */
            if (station->evaluate_element_factor)
            {
                if (station->use_polarised_elements)
                {
                    /* Get pointer to G. */
                    if (oskar_mem_is_matrix(EG->type))
                    {
                        /* Use the memory passed to the function. */
                        G_ptr = EG;
                    }
                    else
                    {
                        /* Use work buffer. */
                        error = oskar_mem_get_pointer(&G_temp, &work->matrix,
                                work->used_matrix, num_points);
                        work->used_matrix += num_points;
                        if (error) return error;
                        G_ptr = &G_temp;
                    }

                    if (!station->element_pattern)
                    {
                        double cos_x, sin_x, cos_y, sin_y;

                        /* Get common dipole orientations. */
                        cos_x = cos(station->orientation_x);
                        sin_x = sin(station->orientation_x);
                        cos_y = cos(station->orientation_y);
                        sin_y = sin(station->orientation_y);

                        /* Evaluate dipole pattern. */
                        error = oskar_evaluate_dipole_pattern(G_ptr, l, m, n,
                                cos_x, sin_x, cos_y, sin_y);
                        if (error) return error;
                    }
                    else
                    {
                        double cos_x, sin_x, cos_y, sin_y;

                        /* Get common dipole orientations.
                         * NOTE: Currently unused! */
                        cos_x = cos(station->orientation_x);
                        sin_x = sin(station->orientation_x);
                        cos_y = cos(station->orientation_y);
                        sin_y = sin(station->orientation_y);

                        /* Evaluate spline pattern. */
                        error = oskar_evaluate_spline_pattern(G_ptr,
                                station->element_pattern, l, m, n,
                                cos_x, sin_x, cos_y, sin_y, work);
                        if (error) return error;
                    }
                }
                else
                {
                    /* Evaluate a taper for a point-like antenna. */
                    /* Use the complex work buffer for G_ptr. */
                    work->used_complex += num_points;

                    /* This may not be required? */
                }
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

        /* Blank sources below the horizon. */
        error = oskar_blank_below_horizon(EG, n);
        if (error) return error;

        /* Release use of work arrays. */
        work->used_complex -= workspace_complex;
        work->used_matrix -= workspace_matrix;
        /*printf("Complex: %d, Matrix: %d\n", work->used_complex,
                work->used_matrix);*/
    }

    return error;
}

#ifdef __cplusplus
}
#endif
