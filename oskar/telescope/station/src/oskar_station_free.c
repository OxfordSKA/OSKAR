/*
 * Copyright (c) 2011-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_free(oskar_Station* model, int* status)
{
    int i = 0, feed = 0, dim = 0;
    if (!model) return;

    /* Free the element data. */
    for (feed = 0; feed < 2; feed++)
    {
        for (dim = 0; dim < 3; dim++)
        {
            oskar_mem_free(model->element_true_enu_metres[feed][dim], status);
            oskar_mem_free(model->element_measured_enu_metres[feed][dim], status);
            oskar_mem_free(model->element_euler_cpu[feed][dim], status);
        }
        oskar_mem_free(model->element_weight[feed], status);
        oskar_mem_free(model->element_cable_length_error[feed], status);
        oskar_mem_free(model->element_gain[feed], status);
        oskar_mem_free(model->element_gain_error[feed], status);
        oskar_mem_free(model->element_phase_offset_rad[feed], status);
        oskar_mem_free(model->element_phase_error_rad[feed], status);
    }
    oskar_mem_free(model->element_types, status);
    oskar_mem_free(model->element_types_cpu, status);
    oskar_mem_free(model->element_mount_types_cpu, status);
    oskar_mem_free(model->permitted_beam_az_rad, status);
    oskar_mem_free(model->permitted_beam_el_rad, status);

    /* Free the noise model. */
    oskar_mem_free(model->noise_freq_hz, status);
    oskar_mem_free(model->noise_rms_jy, status);

    /* Free the gain model. */
    oskar_gains_free(model->gains, status);

    /* Free the HARP data. */
    oskar_mem_free(model->harp_freq_cpu, status);
    for (i = 0; i < model->harp_num_freq; ++i)
    {
        oskar_harp_free(model->harp_data[i]);
    }
    free(model->harp_data);

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
