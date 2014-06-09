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

#include <private_splines.h>
#include <private_element.h>
#include <oskar_element.h>

#include <oskar_binary_tag_index_free.h>
#include <oskar_binary_stream_read.h>
#include <oskar_mem_binary_stream_read.h>

#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

static void read_splines(FILE* fhan, oskar_Splines* splines,
        oskar_BinaryTagIndex** idx, int index, int* status);

void oskar_element_read(oskar_Element* data, int port, double freq_hz,
        const char* filename, int* status)
{
    FILE* fhan = 0;
    oskar_Splines *h_re, *h_im, *v_re, *v_im;
    oskar_BinaryTagIndex* idx = 0;
    int i, n;

    /* Check all inputs. */
    if (!data || !filename  || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check if this frequency has already been set, and get its index if so. */
    n = data->num_freq;
    for (i = 0; i < n; ++i)
    {
        if (fabs(data->freqs_hz[i] - freq_hz) <= freq_hz * DBL_EPSILON)
            break;
    }

    /* Expand arrays to hold data for a new frequency, if needed. */
    if (i >= data->num_freq)
    {
        i = data->num_freq;
        oskar_element_resize_freq_data(data, i + 1, status);
    }

    /* Store the frequency. */
    data->freqs_hz[i] = freq_hz;

    /* Get pointers to surface data based on port number and frequency index. */
    if (port == 1)
    {
        h_re = oskar_element_x_h_re(data, i);
        h_im = oskar_element_x_h_im(data, i);
        v_re = oskar_element_x_v_re(data, i);
        v_im = oskar_element_x_v_im(data, i);
    }
    else if (port == 2)
    {
        h_re = oskar_element_y_h_re(data, i);
        h_im = oskar_element_y_h_im(data, i);
        v_re = oskar_element_y_v_re(data, i);
        v_im = oskar_element_y_v_im(data, i);
    }
    else
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    /* Read data from binary file. */
    fhan = fopen(filename, "rb");

    /* Read data for [h_re], [h_im], [v_re], [v_im] surfaces. */
    read_splines(fhan, h_re, &idx, 0, status);
    read_splines(fhan, h_im, &idx, 1, status);
    read_splines(fhan, v_re, &idx, 2, status);
    read_splines(fhan, v_im, &idx, 3, status);

    /* Close the file. */
    fclose(fhan);

    /* Store the filename. */
    if (port == 1)
    {
        oskar_mem_append_raw(data->filename_x[i], filename, OSKAR_CHAR,
                OSKAR_CPU, 1 + strlen(filename), status);
    }
    else if (port == 2)
    {
        oskar_mem_append_raw(data->filename_y[i], filename, OSKAR_CHAR,
                OSKAR_CPU, 1 + strlen(filename), status);
    }

    /* Free the tag index. */
    oskar_binary_tag_index_free(idx, status);
}

static void read_splines(FILE* fhan, oskar_Splines* splines,
        oskar_BinaryTagIndex** idx, int index, int* status)
{
    unsigned char group;
    group = OSKAR_TAG_GROUP_SPLINE_DATA;
    oskar_binary_stream_read_int(fhan, idx, group,
            OSKAR_SPLINES_TAG_NUM_KNOTS_X_THETA, index,
            &splines->num_knots_x_theta, status);
    oskar_binary_stream_read_int(fhan, idx, group,
            OSKAR_SPLINES_TAG_NUM_KNOTS_Y_PHI, index,
            &splines->num_knots_y_phi, status);
    oskar_mem_binary_stream_read(oskar_splines_knots_x(splines),
            fhan, idx, group, OSKAR_SPLINES_TAG_KNOTS_X_THETA, index, status);
    oskar_mem_binary_stream_read(oskar_splines_knots_y(splines),
            fhan, idx, group, OSKAR_SPLINES_TAG_KNOTS_Y_PHI, index, status);
    oskar_mem_binary_stream_read(oskar_splines_coeff(splines),
            fhan, idx, group, OSKAR_SPLINES_TAG_COEFF, index, status);
    oskar_binary_stream_read_double(fhan, idx, group,
            OSKAR_SPLINES_TAG_SMOOTHING_FACTOR, index,
            &splines->smoothing_factor, status);
}

#ifdef __cplusplus
}
#endif
