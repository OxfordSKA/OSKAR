/*
 * Copyright (c) 2014-2015, The University of Oxford
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

#include <private_element.h>
#include <oskar_element.h>
#include <oskar_binary.h>
#include <oskar_binary_write_mem.h>
#include <oskar_find_closest_match.h>

#ifdef __cplusplus
extern "C" {
#endif

static void write_splines(oskar_Binary* h, const oskar_Splines* splines,
        int index, int* status);

void oskar_element_write(const oskar_Element* data, oskar_Log* log,
        const char* filename, int port, double freq_hz, int* status)
{
    const oskar_Splines *h_re = 0, *h_im = 0, *v_re = 0, *v_im = 0;
    const oskar_Splines *scalar_re = 0, *scalar_im = 0;
    oskar_Binary* h = 0;
    int freq_id;
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
        scalar_re = oskar_element_scalar_re_const(data, freq_id);
        scalar_im = oskar_element_scalar_im_const(data, freq_id);
    }
    else if (port == 1)
    {
        h_re = oskar_element_x_h_re_const(data, freq_id);
        h_im = oskar_element_x_h_im_const(data, freq_id);
        v_re = oskar_element_x_v_re_const(data, freq_id);
        v_im = oskar_element_x_v_im_const(data, freq_id);
    }
    else if (port == 2)
    {
        h_re = oskar_element_y_h_re_const(data, freq_id);
        h_im = oskar_element_y_h_im_const(data, freq_id);
        v_re = oskar_element_y_v_re_const(data, freq_id);
        v_im = oskar_element_y_v_im_const(data, freq_id);
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
    const oskar_Mem *knots_x, *knots_y, *coeff;
    oskar_Mem* temp;
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
