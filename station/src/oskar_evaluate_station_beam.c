/*
 * Copyright (c) 2011, The University of Oxford
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

#include "station/oskar_evaluate_station_beam_scalar.h"
#include "station/oskar_evaluate_station_beam.h"
#include "station/oskar_station_model_location.h"
#include "station/oskar_StationModel.h"
#include "utility/oskar_mem_get_pointer.h"
#include "utility/oskar_mem_realloc.h"
#include "utility/oskar_mem_scale_real.h"
#include "utility/oskar_mem_type_check.h"
#include "utility/oskar_Mem.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_evaluate_station_beam(oskar_Mem* E, const oskar_StationModel* station,
        const double l_beam, const double m_beam, const oskar_Mem* l,
        const oskar_Mem* m, const oskar_Mem* n, oskar_Mem* weights,
        oskar_Device_curand_state* curand_states)
{
    int error;
    oskar_Mem weights_error;

    /* Sanity check on inputs. */
    if (E == NULL || station == NULL || l == NULL || m == NULL || n == NULL ||
            weights == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* NOTE extra fields will have to be added to these check
     * for element pattern data. */

    if (station->coord_units != OSKAR_WAVENUMBERS)
        return OSKAR_ERR_BAD_UNITS;

    if (!E->data || !station->x_weights.data || !station->y_weights.data ||
            !l->data || !m->data || !n->data)
    {
        return OSKAR_ERR_MEMORY_NOT_ALLOCATED;
    }

    /* Check that the relevant memory is on the GPU. */
    if (E->location != OSKAR_LOCATION_GPU ||
            oskar_station_model_location(station) != OSKAR_LOCATION_GPU ||
            weights->location != OSKAR_LOCATION_GPU ||
            l->location != OSKAR_LOCATION_GPU ||
            m->location != OSKAR_LOCATION_GPU ||
            n->location != OSKAR_LOCATION_GPU)
    {
        return OSKAR_ERR_BAD_LOCATION;
    }

    if (l->num_elements != E->num_elements ||
            m->num_elements != E->num_elements ||
            n->num_elements != E->num_elements)
    {
        return OSKAR_ERR_DIMENSION_MISMATCH;
    }

    /* Check the data types. */
    if (oskar_mem_is_real(E->type) || oskar_mem_is_real(weights->type) ||
            oskar_mem_is_complex(l->type) || oskar_mem_is_complex(m->type) ||
            oskar_mem_is_complex(n->type))
        return OSKAR_ERR_BAD_DATA_TYPE;

    /* Resize the weights work array if needed. */
    if (weights->num_elements < 2 * station->num_elements)
    {
        error = oskar_mem_realloc(weights, 2 * station->num_elements);
        if (error) return error;
    }

    /* Non-owned Mem pointer to store the weights error work array. */
    error = oskar_mem_get_pointer(&weights_error, weights,
            station->num_elements, station->num_elements);
    if (error) return error;

    /* No element pattern data. Assume isotropic antenna elements. */
    if (station->element_pattern == NULL && oskar_mem_is_scalar(E->type))
    {
        error = oskar_evaluate_station_beam_scalar(E, station, l_beam, m_beam,
                l, m, n, weights, &weights_error, curand_states);
        if (error) return error;

        if (station->normalise_beam)
        {
            error = oskar_mem_scale_real(E, 1.0/station->num_elements);
            if (error) return error;
        }
    }

    /* Make use of element pattern data. */
    else if (station->element_pattern != NULL && !oskar_mem_is_scalar(E->type))
    {
        return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    }
    else
    {
        return OSKAR_ERR_UNKNOWN;
    }

    return 0;
}

#ifdef __cplusplus
}
#endif
