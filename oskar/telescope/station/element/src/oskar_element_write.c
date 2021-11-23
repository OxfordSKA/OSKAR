/*
 * Copyright (c) 2014-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "binary/oskar_binary.h"
#include "log/oskar_log.h"
#include "math/oskar_find_closest_match.h"
#include "mem/oskar_binary_write_mem.h"
#include "telescope/station/element/private_element.h"
#include "telescope/station/element/oskar_element.h"

#ifdef __cplusplus
extern "C" {
#endif

static void write_splines(oskar_Binary* h, const oskar_Splines* splines,
        int index, int* status);

void oskar_element_write(const oskar_Element* data, const char* filename,
        int port, double freq_hz, oskar_Log* log, int* status)
{
    const oskar_Splines *h_re = 0, *h_im = 0, *v_re = 0, *v_im = 0;
    const oskar_Splines *scalar_re = 0, *scalar_im = 0;
    oskar_Binary* h = 0;
    int freq_id = 0;
    char* log_data = 0;
    size_t log_size = 0;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the frequency ID. */
    freq_id = oskar_find_closest_match_d(freq_hz,
            oskar_element_num_freq(data),
            oskar_element_freqs_hz_const(data));

    /* Get pointers to surface data based on port number and frequency index. */
    if (port == 0)
    {
        scalar_re = data->scalar_re[freq_id];
        scalar_im = data->scalar_im[freq_id];
    }
    else if (port == 1)
    {
        h_re = data->x_h_re[freq_id];
        h_im = data->x_h_im[freq_id];
        v_re = data->x_v_re[freq_id];
        v_im = data->x_v_im[freq_id];
    }
    else if (port == 2)
    {
        h_re = data->y_h_re[freq_id];
        h_im = data->y_h_im[freq_id];
        v_re = data->y_v_re[freq_id];
        v_im = data->y_v_im[freq_id];
    }
    else
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    /* Dump data to a binary file. */
    h = oskar_binary_create(filename, 'w', status);
    if (*status)
    {
        oskar_binary_free(h);
        return;
    }

    /* If log exists, then write it out. */
    log_data = oskar_log_file_data(log, &log_size);
    if (log_data)
    {
        oskar_binary_write(h, OSKAR_CHAR,
                OSKAR_TAG_GROUP_RUN, OSKAR_TAG_RUN_LOG, 0, log_size, log_data,
                status);
        free(log_data);
    }

    if (port == 0)
    {
        /* Write the surface type (scalar). */
        oskar_binary_write_int(h, OSKAR_TAG_GROUP_ELEMENT_DATA,
                OSKAR_ELEMENT_TAG_SURFACE_TYPE, 0,
                OSKAR_ELEMENT_SURFACE_TYPE_SCALAR, status);

        /* Write data for [real], [imag] surfaces. */
        write_splines(h, scalar_re, 0, status);
        write_splines(h, scalar_im, 1, status);
    }
    else
    {
        /* Write the surface type (Ludwig-3). */
        oskar_binary_write_int(h, OSKAR_TAG_GROUP_ELEMENT_DATA,
                OSKAR_ELEMENT_TAG_SURFACE_TYPE, 0,
                OSKAR_ELEMENT_SURFACE_TYPE_LUDWIG_3, status);

        /* Write data for [h_re], [h_im], [v_re], [v_im] surfaces. */
        write_splines(h, h_re, 0, status);
        write_splines(h, h_im, 1, status);
        write_splines(h, v_re, 2, status);
        write_splines(h, v_im, 3, status);
    }

    /* Release the handle. */
    oskar_binary_free(h);
}

static void write_splines(oskar_Binary* h, const oskar_Splines* splines,
        int index, int* status)
{
    const oskar_Mem *knots_x = 0, *knots_y = 0, *coeff = 0;
    oskar_Mem* temp = 0;
    unsigned char group = (unsigned char) OSKAR_TAG_GROUP_SPLINE_DATA;

    if (!splines)
    {
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;
        return;
    }
    if (*status) return;

    knots_x = oskar_splines_knots_x_theta_const(splines);
    knots_y = oskar_splines_knots_y_phi_const(splines);
    coeff   = oskar_splines_coeff_const(splines);
    oskar_binary_write_int(h, group, OSKAR_SPLINES_TAG_NUM_KNOTS_X_THETA,
            index, oskar_splines_num_knots_x_theta(splines), status);
    oskar_binary_write_int(h, group, OSKAR_SPLINES_TAG_NUM_KNOTS_Y_PHI,
            index, oskar_splines_num_knots_y_phi(splines), status);

    /* Write data in double precision. */
    temp = oskar_mem_convert_precision(knots_x, OSKAR_DOUBLE, status);
    oskar_binary_write_mem(h, temp, group,
            OSKAR_SPLINES_TAG_KNOTS_X_THETA, index, 0, status);
    oskar_mem_free(temp, status);
    temp = oskar_mem_convert_precision(knots_y, OSKAR_DOUBLE, status);
    oskar_binary_write_mem(h, temp, group,
            OSKAR_SPLINES_TAG_KNOTS_Y_PHI, index, 0, status);
    oskar_mem_free(temp, status);
    temp = oskar_mem_convert_precision(coeff, OSKAR_DOUBLE, status);
    oskar_binary_write_mem(h, temp, group,
            OSKAR_SPLINES_TAG_COEFF, index, 0, status);
    oskar_mem_free(temp, status);

    /* Write data in single precision. */
    temp = oskar_mem_convert_precision(knots_x, OSKAR_SINGLE, status);
    oskar_binary_write_mem(h, temp, group,
            OSKAR_SPLINES_TAG_KNOTS_X_THETA, index, 0, status);
    oskar_mem_free(temp, status);
    temp = oskar_mem_convert_precision(knots_y, OSKAR_SINGLE, status);
    oskar_binary_write_mem(h, temp, group,
            OSKAR_SPLINES_TAG_KNOTS_Y_PHI, index, 0, status);
    oskar_mem_free(temp, status);
    temp = oskar_mem_convert_precision(coeff, OSKAR_SINGLE, status);
    oskar_binary_write_mem(h, temp, group,
            OSKAR_SPLINES_TAG_COEFF, index, 0, status);
    oskar_mem_free(temp, status);

    oskar_binary_write_double(h, group, OSKAR_SPLINES_TAG_SMOOTHING_FACTOR,
            index, oskar_splines_smoothing_factor(splines), status);
}

#ifdef __cplusplus
}
#endif
