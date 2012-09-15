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
#include "station/oskar_evaluate_station_beam_aperture_array.h"
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
static void check_inputs(oskar_Mem* beam, const oskar_StationModel* station,
        int num_points, const oskar_Mem* x, const oskar_Mem* y,
        const oskar_Mem* z, const oskar_Mem* horizon_mask,
        oskar_WorkStationBeam* work, oskar_Device_curand_state* curand_states,
        int* status);


void oskar_evaluate_station_beam(oskar_Mem* beam,
        const oskar_StationModel* station, double beam_x, double beam_y,
        double beam_z, int num_points, oskar_station_beam_coord_type type,
        const oskar_Mem* x, const oskar_Mem* y, const oskar_Mem* z,
        const oskar_Mem* horizon_mask, oskar_WorkStationBeam* work,
        oskar_Device_curand_state* curand_states, int* status)
{
    check_inputs(beam, station, num_points, x, y, z, horizon_mask, work,
            curand_states, status);
    if (*status) return;

    switch (station->station_type)
    {
        /* Aperture array station */
        case OSKAR_STATION_TYPE_AA:
        {
            if (type != HORIZONTAL_XYZ)
            {
                *status = OSKAR_ERR_SETTINGS_TELESCOPE;
                return;
            }
            oskar_evaluate_station_beam_aperture_array(beam, station, beam_x,
                    beam_y, beam_z, num_points, x, y, z, work, curand_states,
                    status);
            if (*status) return;
            break;
        }

        /* Circular Gaussian beam */
        case OSKAR_STATION_TYPE_GAUSSIAN_BEAM:
        {
            if (type != PHASE_CENTRE_XYZ)
            {
                *status = OSKAR_ERR_SETTINGS_TELESCOPE;
                return;
            }
            oskar_evaluate_station_beam_gaussian(beam, num_points, x, y,
                    station->gaussian_beam_fwhm_deg, status);
            if (*status) return;

            /* Blank (zero) sources below the horizon. */
            *status = oskar_blank_below_horizon(beam, horizon_mask, num_points);
            if (*status) return;
            break;
        }

        /* Dish */
        case OSKAR_STATION_TYPE_DISH:
        {
            *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
            return;
        }

        default:
        {
            *status = OSKAR_ERR_SETTINGS_INTERFEROMETER;
            return;
        }

    };
}



static void check_inputs(oskar_Mem* beam, const oskar_StationModel* station,
        int num_points, const oskar_Mem* x, const oskar_Mem* y,
        const oskar_Mem* z, const oskar_Mem* horizon_mask,
        oskar_WorkStationBeam* work, oskar_Device_curand_state* curand_states,
        int* status)
{
    /* Sanity check on inputs. */
    if (!beam || !station || !x || !y || !z || !horizon_mask || !work ||
            !curand_states || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    if (*status) return;

    /* Check the coordinate units. */
    if (station->coord_units != OSKAR_RADIANS)
    {
        *status = OSKAR_ERR_BAD_UNITS;
        return;
    }

    /* Check that there is memory available. */
    if (!beam->data || !x->data || !y->data || !z->data)
    {
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;
        return;
    }

    /* Check that the relevant memory is on the GPU. */
    if (oskar_station_model_location(station) != OSKAR_LOCATION_GPU ||
            beam->location != OSKAR_LOCATION_GPU ||
            x->location != OSKAR_LOCATION_GPU ||
            y->location != OSKAR_LOCATION_GPU ||
            z->location != OSKAR_LOCATION_GPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    /* Check that the array sizes are OK. */
    if (beam->num_elements < num_points || x->num_elements < num_points ||
            y->num_elements < num_points || z->num_elements < num_points)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Check the data types. */
    if (oskar_mem_is_real(beam->type) || oskar_mem_is_complex(x->type) ||
            oskar_mem_is_complex(y->type) || oskar_mem_is_complex(x->type))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
}


#ifdef __cplusplus
}
#endif
