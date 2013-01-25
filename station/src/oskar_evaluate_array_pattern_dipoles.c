/*
 * Copyright (c) 2013, The University of Oxford
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

#include "station/oskar_evaluate_array_pattern_dipoles.h"
#include "station/oskar_evaluate_array_pattern_dipoles_cuda.h"
#include "station/oskar_evaluate_element_weights.h"
#include "station/oskar_station_model_location.h"
#include "station/oskar_station_model_type.h"
#include "utility/oskar_cuda_check_error.h"
#include "utility/oskar_mem_realloc.h"
#include "utility/oskar_mem_type_check.h"

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_evaluate_array_pattern_dipoles(oskar_Mem* beam,
        const oskar_StationModel* station, double beam_x, double beam_y,
        double beam_z, int num_points, const oskar_Mem* x, const oskar_Mem* y,
        const oskar_Mem* z, oskar_Mem* weights, oskar_Mem* weights_error,
        oskar_CurandState* curand_state, int* status)
{
    int type, location;

    /* Check all inputs. */
    if (!beam || !station || !x || !y || !z || !weights || !weights_error ||
            !curand_state || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get meta-data. */
    location = oskar_station_model_location(station);
    type = oskar_station_model_type(station);

    /* Check data are co-located. */
    if (beam->location != location ||
            x->location != location ||
            y->location != location ||
            z->location != location ||
            weights->location != location ||
            weights_error->location != location)
        *status = OSKAR_ERR_LOCATION_MISMATCH;

    /* Check that the antenna coordinates are in radians. */
    if (station->coord_units != OSKAR_RADIANS)
        *status = OSKAR_ERR_BAD_UNITS;

    /* Check for correct data types. */
    if (!oskar_mem_is_complex(beam->type) ||
            !oskar_mem_is_complex(weights->type) ||
            !oskar_mem_is_matrix(beam->type) ||
            oskar_mem_is_matrix(weights->type))
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    if (x->type != type || y->type != type || z->type != type ||
            oskar_mem_base_type(beam->type) != type ||
            oskar_mem_base_type(weights->type) != type)
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Resize output array if required. */
    if (beam->num_elements < num_points)
        oskar_mem_realloc(beam, num_points, status);

    /* Generate the beamforming weights. */
    oskar_evaluate_element_weights(weights, weights_error, station,
            beam_x, beam_y, beam_z, curand_state, status);

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check for data in GPU memory. */
    if (location == OSKAR_LOCATION_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (type == OSKAR_DOUBLE)
        {
            oskar_evaluate_array_pattern_dipoles_cuda_d(station->num_elements,
                    (const double*)station->x_signal.data,
                    (const double*)station->y_signal.data,
                    (const double*)station->z_signal.data,
                    (const double*)station->cos_orientation_x.data,
                    (const double*)station->sin_orientation_x.data,
                    (const double*)station->cos_orientation_y.data,
                    (const double*)station->sin_orientation_y.data,
                    (const double2*)weights->data, num_points,
                    (const double*)x->data,
                    (const double*)y->data,
                    (const double*)z->data,
                    (double4c*)beam->data);
            oskar_cuda_check_error(status);
        }
        else if (type == OSKAR_SINGLE)
        {
            oskar_evaluate_array_pattern_dipoles_cuda_f(station->num_elements,
                    (const float*)station->x_signal.data,
                    (const float*)station->y_signal.data,
                    (const float*)station->z_signal.data,
                    (const float*)station->cos_orientation_x.data,
                    (const float*)station->sin_orientation_x.data,
                    (const float*)station->cos_orientation_y.data,
                    (const float*)station->sin_orientation_y.data,
                    (const float2*)weights->data, num_points,
                    (const float*)x->data,
                    (const float*)y->data,
                    (const float*)z->data,
                    (float4c*)beam->data);
            oskar_cuda_check_error(status);
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else
    {
        *status = OSKAR_ERR_BAD_LOCATION;
    }
}

#ifdef __cplusplus
}
#endif
