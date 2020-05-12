/*
 * Copyright (c) 2013-2020, The University of Oxford
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

#define COPY_OR_CREATE(MEM)\
    if (src->MEM && !dst->MEM)\
        dst->MEM = oskar_mem_create_copy(src->MEM, location, status);\
    else oskar_mem_copy(dst->MEM, src->MEM, status);


oskar_Station* oskar_station_create_copy(const oskar_Station* src,
        int location, int* status)
{
    int i, feed, dim;
    oskar_Station* dst = 0;

    /* Check if safe to proceed. */
    if (*status || !src) return dst;

    /* Create the new model. */
    dst = oskar_station_create(oskar_station_precision(src), location,
            oskar_station_num_elements(src), status);

    /* Set meta-data. */
    dst->unique_id = src->unique_id;
    dst->precision = src->precision;
    dst->mem_location = location;

    /* Copy common station parameters. */
    dst->station_type = src->station_type;
    dst->normalise_final_beam = src->normalise_final_beam;
    dst->offset_ecef[0] = src->offset_ecef[0];
    dst->offset_ecef[1] = src->offset_ecef[1];
    dst->offset_ecef[2] = src->offset_ecef[2];
    dst->lon_rad = src->lon_rad;
    dst->lat_rad = src->lat_rad;
    dst->alt_metres = src->alt_metres;
    dst->pm_x_rad = src->pm_x_rad;
    dst->pm_y_rad = src->pm_y_rad;
    dst->beam_lon_rad = src->beam_lon_rad;
    dst->beam_lat_rad = src->beam_lat_rad;
    dst->beam_coord_type = src->beam_coord_type;
    oskar_mem_copy(dst->noise_freq_hz, src->noise_freq_hz, status);
    oskar_mem_copy(dst->noise_rms_jy, src->noise_rms_jy, status);

    /* Copy aperture array data, except num_element_types (done later). */
    dst->identical_children = src->identical_children;
    dst->num_elements = src->num_elements;
    dst->normalise_array_pattern = src->normalise_array_pattern;
    dst->normalise_element_pattern = src->normalise_element_pattern;
    dst->enable_array_pattern = src->enable_array_pattern;
    dst->common_element_orientation = src->common_element_orientation;
    dst->common_pol_beams = src->common_pol_beams;
    dst->array_is_3d = src->array_is_3d;
    dst->apply_element_errors = src->apply_element_errors;
    dst->apply_element_weight = src->apply_element_weight;
    dst->seed_time_variable_errors = src->seed_time_variable_errors;
    dst->num_permitted_beams = src->num_permitted_beams;

    /* Copy Gaussian station beam data. */
    dst->gaussian_beam_fwhm_rad = src->gaussian_beam_fwhm_rad;
    dst->gaussian_beam_reference_freq_hz = src->gaussian_beam_reference_freq_hz;

    /* Copy arrays. */
    for (feed = 0; feed < 2; feed++)
    {
        for (dim = 0; dim < 3; dim++)
        {
            COPY_OR_CREATE(element_true_enu_metres[feed][dim])
            COPY_OR_CREATE(element_measured_enu_metres[feed][dim])
            COPY_OR_CREATE(element_euler_cpu[feed][dim])
        }
        COPY_OR_CREATE(element_weight[feed])
        COPY_OR_CREATE(element_cable_length_error[feed])
        COPY_OR_CREATE(element_gain[feed])
        COPY_OR_CREATE(element_gain_error[feed])
        COPY_OR_CREATE(element_phase_offset_rad[feed])
        COPY_OR_CREATE(element_phase_error_rad[feed])
    }
    oskar_mem_copy(dst->element_types, src->element_types, status);
    oskar_mem_copy(dst->element_types_cpu, src->element_types_cpu, status);
    oskar_mem_copy(dst->element_mount_types_cpu, src->element_mount_types_cpu, status);
    oskar_mem_copy(dst->permitted_beam_az_rad, src->permitted_beam_az_rad, status);
    oskar_mem_copy(dst->permitted_beam_el_rad, src->permitted_beam_el_rad, status);

    /* Copy element models, if set. */
    if (oskar_station_has_element(src))
    {
        /* Ensure enough space for element model data. */
        oskar_station_resize_element_types(dst, src->num_element_types,
                status);

        /* Copy the element model data. */
        for (i = 0; i < src->num_element_types; ++i)
        {
            oskar_element_copy(dst->element[i], src->element[i], status);
        }
    }

    /* Recursively copy child stations. */
    if (oskar_station_has_child(src))
    {
        dst->child = (oskar_Station**) calloc(
                src->num_elements, sizeof(oskar_Station*));
        if (!dst->child)
        {
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
            return dst;
        }

        for (i = 0; i < src->num_elements; ++i)
        {
            dst->child[i] = oskar_station_create_copy(
                    oskar_station_child_const(src, i), location, status);
        }
    }

    return dst;
}

#ifdef __cplusplus
}
#endif
