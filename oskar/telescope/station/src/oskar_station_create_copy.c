/*
 * Copyright (c) 2013-2015, The University of Oxford
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

#include "telescope/station/element/oskar_element_copy.h"

#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Station* oskar_station_create_copy(const oskar_Station* src,
        int location, int* status)
{
    int i = 0;
    oskar_Station* model = 0;

    /* Check if safe to proceed. */
    if (*status) return model;

    /* Create the new model. */
    model = oskar_station_create(oskar_station_precision(src), location,
            oskar_station_num_elements(src), status);

    /* Set meta-data. */
    model->unique_id = src->unique_id;
    model->precision = src->precision;
    model->mem_location = location;

    /* Copy common station parameters. */
    model->station_type = src->station_type;
    model->normalise_final_beam = src->normalise_final_beam;
    model->lon_rad = src->lon_rad;
    model->lat_rad = src->lat_rad;
    model->alt_metres = src->alt_metres;
    model->pm_x_rad = src->pm_x_rad;
    model->pm_y_rad = src->pm_y_rad;
    model->beam_lon_rad = src->beam_lon_rad;
    model->beam_lat_rad = src->beam_lat_rad;
    model->beam_coord_type = src->beam_coord_type;
    oskar_mem_copy(model->noise_freq_hz, src->noise_freq_hz, status);
    oskar_mem_copy(model->noise_rms_jy, src->noise_rms_jy, status);

    /* Copy aperture array data, except num_element_types (done later). */
    model->identical_children = src->identical_children;
    model->num_elements = src->num_elements;
    model->normalise_array_pattern = src->normalise_array_pattern;
    model->enable_array_pattern = src->enable_array_pattern;
    model->common_element_orientation = src->common_element_orientation;
    model->array_is_3d = src->array_is_3d;
    model->apply_element_errors = src->apply_element_errors;
    model->apply_element_weight = src->apply_element_weight;
    model->seed_time_variable_errors = src->seed_time_variable_errors;
    model->num_permitted_beams = src->num_permitted_beams;

    /* Copy Gaussian station beam data. */
    model->gaussian_beam_fwhm_rad = src->gaussian_beam_fwhm_rad;
    model->gaussian_beam_reference_freq_hz = src->gaussian_beam_reference_freq_hz;

    /* Copy memory blocks. */
    oskar_mem_copy(model->element_true_x_enu_metres,
            src->element_true_x_enu_metres, status);
    oskar_mem_copy(model->element_true_y_enu_metres,
            src->element_true_y_enu_metres, status);
    oskar_mem_copy(model->element_true_z_enu_metres,
            src->element_true_z_enu_metres, status);
    oskar_mem_copy(model->element_measured_x_enu_metres,
            src->element_measured_x_enu_metres, status);
    oskar_mem_copy(model->element_measured_y_enu_metres,
            src->element_measured_y_enu_metres, status);
    oskar_mem_copy(model->element_measured_z_enu_metres,
            src->element_measured_z_enu_metres, status);
    oskar_mem_copy(model->element_weight, src->element_weight, status);
    oskar_mem_copy(model->element_gain, src->element_gain, status);
    oskar_mem_copy(model->element_gain_error, src->element_gain_error, status);
    oskar_mem_copy(model->element_phase_offset_rad,
            src->element_phase_offset_rad, status);
    oskar_mem_copy(model->element_phase_error_rad,
            src->element_phase_error_rad, status);
    oskar_mem_copy(model->element_x_alpha_cpu,
            src->element_x_alpha_cpu, status);
    oskar_mem_copy(model->element_x_beta_cpu,
            src->element_x_beta_cpu, status);
    oskar_mem_copy(model->element_x_gamma_cpu,
            src->element_x_gamma_cpu, status);
    oskar_mem_copy(model->element_y_alpha_cpu,
            src->element_y_alpha_cpu, status);
    oskar_mem_copy(model->element_y_beta_cpu,
            src->element_y_beta_cpu, status);
    oskar_mem_copy(model->element_y_gamma_cpu,
            src->element_y_gamma_cpu, status);
    oskar_mem_copy(model->element_types, src->element_types, status);
    oskar_mem_copy(model->element_types_cpu, src->element_types_cpu, status);
    oskar_mem_copy(model->element_mount_types_cpu, src->element_mount_types_cpu, status);
    oskar_mem_copy(model->permitted_beam_az_rad, src->permitted_beam_az_rad, status);
    oskar_mem_copy(model->permitted_beam_el_rad, src->permitted_beam_el_rad, status);

    /* Copy element models, if set. */
    if (oskar_station_has_element(src))
    {
        /* Ensure enough space for element model data. */
        oskar_station_resize_element_types(model, src->num_element_types,
                status);

        /* Copy the element model data. */
        for (i = 0; i < src->num_element_types; ++i)
        {
            oskar_element_copy(model->element[i], src->element[i], status);
        }
    }

    /* Recursively copy child stations. */
    if (oskar_station_has_child(src))
    {
        model->child = malloc(src->num_elements * sizeof(oskar_Station*));
        if (!model->child)
        {
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
            return model;
        }

        for (i = 0; i < src->num_elements; ++i)
        {
            model->child[i] = oskar_station_create_copy(
                    oskar_station_child_const(src, i), location, status);
        }
    }

    return model;
}

#ifdef __cplusplus
}
#endif
