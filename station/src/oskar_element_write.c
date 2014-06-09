/*
 * Copyright (c) 2014, The University of Oxford
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

#include <oskar_binary_stream_write_header.h>
#include <oskar_binary_stream_write.h>
#include <oskar_mem_binary_stream_write.h>
#include <oskar_find_closest_match.h>

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

static void write_splines(FILE* fhan, const oskar_Splines* splines,
        int index, int* status);

void oskar_element_write(const oskar_Element* data, int port, double freq_hz,
        const char* filename, int* status)
{
    FILE* fhan = 0;
    const oskar_Splines *h_re, *h_im, *v_re, *v_im;
    int freq_id;

    /* Check all inputs. */
    if (!data || !filename  || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the frequency ID. */
    freq_id = oskar_find_closest_match_d(freq_hz,
            oskar_element_num_freq(data),
            oskar_element_freqs_hz(data));

    /* Get pointers based on port number. */
    if (port == 1)
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

    /* Check that the data exist. */
    if (!h_re || !h_im || !v_re || !v_im)
    {
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;
        return;
    }

    /* Dump data to a binary file. */
    fhan = fopen(filename, "wb");
    oskar_binary_stream_write_header(fhan, status);

    /* Write data for [h_re], [h_im], [v_re], [v_im] surfaces. */
    write_splines(fhan, h_re, 0, status);
    write_splines(fhan, h_im, 1, status);
    write_splines(fhan, v_re, 2, status);
    write_splines(fhan, v_im, 3, status);

    /* Close the file. */
    fclose(fhan);
}

static void write_splines(FILE* fhan, const oskar_Splines* splines,
        int index, int* status)
{
    const oskar_Mem *knots_x, *knots_y, *coeff;
    oskar_Mem* temp;
    unsigned char group;
    group = OSKAR_TAG_GROUP_SPLINE_DATA;
    knots_x = oskar_splines_knots_x_theta_const(splines);
    knots_y = oskar_splines_knots_y_phi_const(splines);
    coeff   = oskar_splines_coeff_const(splines);
    oskar_binary_stream_write_int(fhan, group,
            OSKAR_SPLINES_TAG_NUM_KNOTS_X_THETA, index,
            oskar_splines_num_knots_x_theta(splines), status);
    oskar_binary_stream_write_int(fhan, group,
            OSKAR_SPLINES_TAG_NUM_KNOTS_Y_PHI, index,
            oskar_splines_num_knots_y_phi(splines), status);

    /* Write data in double precision. */
    temp = oskar_mem_convert_precision(knots_x, OSKAR_DOUBLE, status);
    oskar_mem_binary_stream_write(temp, fhan, group,
            OSKAR_SPLINES_TAG_KNOTS_X_THETA, index, 0, status);
    oskar_mem_free(temp, status);
    temp = oskar_mem_convert_precision(knots_y, OSKAR_DOUBLE, status);
    oskar_mem_binary_stream_write(temp, fhan, group,
            OSKAR_SPLINES_TAG_KNOTS_Y_PHI, index, 0, status);
    oskar_mem_free(temp, status);
    temp = oskar_mem_convert_precision(coeff, OSKAR_DOUBLE, status);
    oskar_mem_binary_stream_write(temp, fhan, group,
            OSKAR_SPLINES_TAG_COEFF, index, 0, status);
    oskar_mem_free(temp, status);

    /* Write data in single precision. */
    temp = oskar_mem_convert_precision(knots_x, OSKAR_SINGLE, status);
    oskar_mem_binary_stream_write(temp, fhan, group,
            OSKAR_SPLINES_TAG_KNOTS_X_THETA, index, 0, status);
    oskar_mem_free(temp, status);
    temp = oskar_mem_convert_precision(knots_y, OSKAR_SINGLE, status);
    oskar_mem_binary_stream_write(temp, fhan, group,
            OSKAR_SPLINES_TAG_KNOTS_Y_PHI, index, 0, status);
    oskar_mem_free(temp, status);
    temp = oskar_mem_convert_precision(coeff, OSKAR_SINGLE, status);
    oskar_mem_binary_stream_write(temp, fhan, group,
            OSKAR_SPLINES_TAG_COEFF, index, 0, status);
    oskar_mem_free(temp, status);

    oskar_binary_stream_write_double(fhan, group,
            OSKAR_SPLINES_TAG_SMOOTHING_FACTOR, index,
            oskar_splines_smoothing_factor(splines), status);
}

#ifdef __cplusplus
}
#endif
