/*
 * Copyright (c) 2011-2015, The University of Oxford
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

#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station.h"
#include "math/oskar_cmath.h"
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Station* oskar_station_create(int type, int location, int num_elements,
        int* status)
{
    oskar_Station* model;

    /* Check the type and location. */
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
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
    model->unique_id = 0;
    model->precision = type;
    model->mem_location = location;

    /* Initialise the memory. */
    model->element_true_x_enu_metres =
            oskar_mem_create(type, location, num_elements, status);
    model->element_true_y_enu_metres =
            oskar_mem_create(type, location, num_elements, status);
    model->element_true_z_enu_metres =
            oskar_mem_create(type, location, num_elements, status);
    model->element_measured_x_enu_metres =
            oskar_mem_create(type, location, num_elements, status);
    model->element_measured_y_enu_metres =
            oskar_mem_create(type, location, num_elements, status);
    model->element_measured_z_enu_metres =
            oskar_mem_create(type, location, num_elements, status);
    model->element_weight =
            oskar_mem_create(type | OSKAR_COMPLEX, location, num_elements, status);
    model->element_gain =
            oskar_mem_create(type, location, num_elements, status);
    model->element_gain_error =
            oskar_mem_create(type, location, num_elements, status);
    model->element_phase_offset_rad =
            oskar_mem_create(type, location, num_elements, status);
    model->element_phase_error_rad =
            oskar_mem_create(type, location, num_elements, status);
    model->element_x_alpha_cpu =
            oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, num_elements, status);
    model->element_x_beta_cpu =
            oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, num_elements, status);
    model->element_x_gamma_cpu =
            oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, num_elements, status);
    model->element_y_alpha_cpu =
            oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, num_elements, status);
    model->element_y_beta_cpu =
            oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, num_elements, status);
    model->element_y_gamma_cpu =
            oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, num_elements, status);
    model->element_types =
            oskar_mem_create(OSKAR_INT, location, num_elements, status);
    model->element_types_cpu =
            oskar_mem_create(OSKAR_INT, OSKAR_CPU, num_elements, status);
    model->element_mount_types_cpu =
            oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, num_elements, status);
    model->permitted_beam_az_rad =
            oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    model->permitted_beam_el_rad =
            oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);

    /* Initialise common data. */
    model->station_type = OSKAR_STATION_TYPE_AA;
    model->normalise_final_beam = OSKAR_TRUE;
    model->lon_rad = 0.0;
    model->lat_rad = 0.0;
    model->alt_metres = 0.0;
    model->pm_x_rad = 0.0;
    model->pm_y_rad = 0.0;
    model->beam_lon_rad = 0.0;
    model->beam_lat_rad = 0.0;
    model->beam_coord_type = OSKAR_SPHERICAL_TYPE_EQUATORIAL;
    model->noise_freq_hz = oskar_mem_create(type, OSKAR_CPU, 0, status);
    model->noise_rms_jy = oskar_mem_create(type, OSKAR_CPU, 0, status);

    /* Initialise Gaussian beam station data. */
    model->gaussian_beam_fwhm_rad = 0.0;
    model->gaussian_beam_reference_freq_hz = 0.0;

    /* Initialise aperture array data. */
    model->identical_children = OSKAR_FALSE;
    model->num_elements = num_elements;
    model->num_element_types = 0;
    model->normalise_array_pattern = OSKAR_FALSE;
    model->enable_array_pattern = OSKAR_TRUE;
    model->common_element_orientation = OSKAR_TRUE;
    model->array_is_3d = OSKAR_FALSE;
    model->apply_element_errors = OSKAR_FALSE;
    model->apply_element_weight = OSKAR_FALSE;
    model->seed_time_variable_errors = 1;
    model->child = 0;
    model->element = 0;
    model->num_permitted_beams = 0;
    if (num_elements > 0)
        memset(oskar_mem_void(model->element_mount_types_cpu), 'F',
                num_elements);

    /* Return pointer to station model. */
    return model;
}

#ifdef __cplusplus
}
#endif
