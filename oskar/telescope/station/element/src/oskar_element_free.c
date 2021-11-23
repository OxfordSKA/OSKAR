/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/element/private_element.h"
#include "telescope/station/element/oskar_element.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_element_free(oskar_Element* data, int* status)
{
    int i = 0;
    if (!data) return;

    /* Free the memory contents. */
    for (i = 0; i < data->num_freq; ++i)
    {
        oskar_mem_free(data->filename_x[i], status);
        oskar_mem_free(data->filename_y[i], status);
        oskar_mem_free(data->filename_scalar[i], status);
        oskar_splines_free(data->x_v_re[i], status);
        oskar_splines_free(data->x_v_im[i], status);
        oskar_splines_free(data->x_h_re[i], status);
        oskar_splines_free(data->x_h_im[i], status);
        oskar_splines_free(data->y_v_re[i], status);
        oskar_splines_free(data->y_v_im[i], status);
        oskar_splines_free(data->y_h_re[i], status);
        oskar_splines_free(data->y_h_im[i], status);
        oskar_splines_free(data->scalar_re[i], status);
        oskar_splines_free(data->scalar_im[i], status);
        oskar_mem_free(data->sph_wave[i], status);
    }
    free(data->freqs_hz);
    free(data->l_max);
    free(data->common_phi_coords);
    free(data->filename_x);
    free(data->filename_y);
    free(data->filename_scalar);
    free(data->x_h_re);
    free(data->x_h_im);
    free(data->x_v_re);
    free(data->x_v_im);
    free(data->y_h_re);
    free(data->y_h_im);
    free(data->y_v_re);
    free(data->y_v_im);
    free(data->scalar_re);
    free(data->scalar_im);
    free(data->sph_wave);

    /* Free the structure itself. */
    free(data);
}

#ifdef __cplusplus
}
#endif
