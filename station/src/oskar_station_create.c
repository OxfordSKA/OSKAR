/*
 * Copyright (c) 2011-2014, The University of Oxford
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

#include <private_station.h>
#include <oskar_station.h>

#include <oskar_system_noise_model_create.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
extern "C" {
#endif

oskar_Station* oskar_station_create(int type, int location, int num_elements,
        int* status)
{
    oskar_Station* model;

    /* Check all inputs. */
    if (!status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Check the type and location. */
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return 0;
    }
    if (location != OSKAR_LOCATION_CPU && location != OSKAR_LOCATION_GPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return 0;
    }

    /* Allocate and initialise a station model structure. */
    model = (oskar_Station*) malloc(sizeof(oskar_Station));
    if (!model)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        return 0;
    }

    /* Initialise station meta data. */
    model->precision = type;
    model->location = location;

    /* Initialise the memory. */
    model->x_signal = oskar_mem_create(type, location, num_elements, status);
    model->y_signal = oskar_mem_create(type, location, num_elements, status);
    model->z_signal = oskar_mem_create(type, location, num_elements, status);
    model->x_weights = oskar_mem_create(type, location, num_elements, status);
    model->y_weights = oskar_mem_create(type, location, num_elements, status);
    model->z_weights = oskar_mem_create(type, location, num_elements, status);
    model->weight = oskar_mem_create(type | OSKAR_COMPLEX, location, num_elements, status);
    model->gain = oskar_mem_create(type, location, num_elements, status);
    model->gain_error = oskar_mem_create(type, location, num_elements, status);
    model->phase_offset = oskar_mem_create(type, location, num_elements, status);
    model->phase_error = oskar_mem_create(type, location, num_elements, status);
    model->orientation_x_cpu = oskar_mem_create(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_elements, status);
    model->orientation_y_cpu = oskar_mem_create(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_elements, status);
    model->element_types = oskar_mem_create(OSKAR_INT, location, num_elements, status);
    model->element_types_cpu = oskar_mem_create(OSKAR_INT, OSKAR_LOCATION_CPU, num_elements, status);
    model->permitted_beam_az = oskar_mem_create(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, 0, status);
    model->permitted_beam_el = oskar_mem_create(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, 0, status);

    /* Initialise common data. */
    model->station_type = OSKAR_STATION_TYPE_AA;
    model->normalise_final_beam = OSKAR_FALSE;
    model->longitude_rad = 0.0;
    model->latitude_rad = 0.0;
    model->altitude_m = 0.0;
    model->beam_longitude_rad = 0.0;
    model->beam_latitude_rad = 0.0;
    model->beam_coord_type = OSKAR_SPHERICAL_TYPE_EQUATORIAL;
    model->noise = oskar_system_noise_model_create(type, location, status);

    /* Initialise Gaussian beam station data. */
    model->gaussian_beam_fwhm_rad = 0.0;

    /* Initialise aperture array data. */
    model->identical_children = OSKAR_FALSE;
    model->num_elements = num_elements;
    model->num_element_types = 0;
    model->use_polarised_elements = OSKAR_TRUE;
    model->normalise_array_pattern = OSKAR_FALSE;
    model->enable_array_pattern = OSKAR_TRUE;
    model->common_element_orientation = OSKAR_TRUE;
    model->array_is_3d = OSKAR_FALSE;
    model->apply_element_errors = OSKAR_FALSE;
    model->apply_element_weight = OSKAR_FALSE;
    model->nominal_orientation_x = M_PI / 2.0;
    model->nominal_orientation_y = 0.0;
    model->child = 0;
    model->element_pattern = 0;
    model->num_permitted_beams = 0;

    /* Return pointer to station model. */
    return model;
}

#ifdef __cplusplus
}
#endif
