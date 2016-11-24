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

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_free(oskar_Station* model, int* status)
{
    int i = 0;
    if (!model) return;

    /* Free the element data. */
    oskar_mem_free(model->element_true_x_enu_metres, status);
    oskar_mem_free(model->element_true_y_enu_metres, status);
    oskar_mem_free(model->element_true_z_enu_metres, status);
    oskar_mem_free(model->element_measured_x_enu_metres, status);
    oskar_mem_free(model->element_measured_y_enu_metres, status);
    oskar_mem_free(model->element_measured_z_enu_metres, status);
    oskar_mem_free(model->element_weight, status);
    oskar_mem_free(model->element_gain, status);
    oskar_mem_free(model->element_gain_error, status);
    oskar_mem_free(model->element_phase_offset_rad, status);
    oskar_mem_free(model->element_phase_error_rad, status);
    oskar_mem_free(model->element_x_alpha_cpu, status);
    oskar_mem_free(model->element_x_beta_cpu, status);
    oskar_mem_free(model->element_x_gamma_cpu, status);
    oskar_mem_free(model->element_y_alpha_cpu, status);
    oskar_mem_free(model->element_y_beta_cpu, status);
    oskar_mem_free(model->element_y_gamma_cpu, status);
    oskar_mem_free(model->element_types, status);
    oskar_mem_free(model->element_types_cpu, status);
    oskar_mem_free(model->element_mount_types_cpu, status);
    oskar_mem_free(model->permitted_beam_az_rad, status);
    oskar_mem_free(model->permitted_beam_el_rad, status);

    /* Free the noise model. */
    oskar_mem_free(model->noise_freq_hz, status);
    oskar_mem_free(model->noise_rms_jy, status);

    /* Free the element pattern data if it exists. */
    if (oskar_station_has_element(model))
    {
        for (i = 0; i < model->num_element_types; ++i)
        {
            oskar_element_free(oskar_station_element(model, i), status);
        }

        /* Free the element model handle array. */
        free(model->element);
    }

    /* Recursively free the child stations. */
    if (oskar_station_has_child(model))
    {
        for (i = 0; i < model->num_elements; ++i)
        {
            oskar_station_free(oskar_station_child(model, i), status);
        }

        /* Free the child station handle array. */
        free(model->child);
    }

    /* Free the structure itself. */
    free(model);
}

#ifdef __cplusplus
}
#endif
