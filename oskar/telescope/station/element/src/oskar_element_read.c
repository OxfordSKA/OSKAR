/*
 * Copyright (c) 2014-2019, The University of Oxford
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

#include "splines/private_splines.h"
#include "telescope/station/element/private_element.h"
#include "telescope/station/element/oskar_element.h"
#include "mem/oskar_binary_read_mem.h"
#include "binary/oskar_binary.h"

#include <string.h>
#include <float.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

static void read_splines(oskar_Element* data, oskar_Binary* h,
        oskar_Splines** splines_ptr, int index, int* status);

void oskar_element_read(oskar_Element* data, const char* filename,
        int feed, double freq_hz, int* status)
{
    oskar_Splines **h_re = 0, **h_im = 0, **v_re = 0, **v_im = 0;
    oskar_Splines **scalar_re = 0, **scalar_im = 0;
    oskar_Mem **filename_ptr = 0;
    oskar_Binary* h = 0;
    int i, n, surface_type = -1;
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

    /* Get pointers to surface data based on feed number and frequency index. */
    if (feed == 0)
    {
        scalar_re = &data->scalar_re[i];
        scalar_im = &data->scalar_im[i];
        filename_ptr = &data->filename_scalar[i];
    }
    else if (feed == 1)
    {
        h_re = &data->x_h_re[i];
        h_im = &data->x_h_im[i];
        v_re = &data->x_v_re[i];
        v_im = &data->x_v_im[i];
        filename_ptr = &data->filename_x[i];
    }
    else if (feed == 2)
    {
        h_re = &data->y_h_re[i];
        h_im = &data->y_h_im[i];
        v_re = &data->y_v_re[i];
        v_im = &data->y_v_im[i];
        filename_ptr = &data->filename_y[i];
    }
    else
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    /* Create the handle. */
    h = oskar_binary_create(filename, 'r', status);
    if (*status)
    {
        oskar_binary_free(h);
        return;
    }

    /* Get the surface type. */
    oskar_binary_read_int(h, OSKAR_TAG_GROUP_ELEMENT_DATA,
            OSKAR_ELEMENT_TAG_SURFACE_TYPE, 0, &surface_type, status);
    if (*status)
    {
        oskar_binary_free(h);
        return;
    }

    if (feed == 0)
    {
        /* Check the surface type (scalar). */
        if (surface_type != OSKAR_ELEMENT_SURFACE_TYPE_SCALAR)
            *status = OSKAR_ERR_INVALID_ARGUMENT;

        /* Read data for [real], [imag] surfaces. */
        read_splines(data, h, scalar_re, 0, status);
        read_splines(data, h, scalar_im, 1, status);
    }
    else
    {
        /* Check the surface type (Ludwig-3). */
        if (surface_type != OSKAR_ELEMENT_SURFACE_TYPE_LUDWIG_3)
            *status = OSKAR_ERR_INVALID_ARGUMENT;

        /* Read data for [h_re], [h_im], [v_re], [v_im] surfaces. */
        read_splines(data, h, h_re, 0, status);
        read_splines(data, h, h_im, 1, status);
        read_splines(data, h, v_re, 2, status);
        read_splines(data, h, v_im, 3, status);
    }

    /* Store the filename. */
    if (!*filename_ptr)
        *filename_ptr = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 0, status);
    oskar_mem_append_raw(*filename_ptr, filename, OSKAR_CHAR,
            OSKAR_CPU, 1 + strlen(filename), status);

    /* Release the handle. */
    oskar_binary_free(h);
}

static void read_splines(oskar_Element* data, oskar_Binary* h,
        oskar_Splines** splines_ptr, int index, int* status)
{
    if (*status) return;
    const unsigned char group = (unsigned char) OSKAR_TAG_GROUP_SPLINE_DATA;
    if (!*splines_ptr)
        *splines_ptr = oskar_splines_create(oskar_element_precision(data),
                oskar_element_mem_location(data), status);
    if (*status) return;
    oskar_Splines* splines = *splines_ptr;

    oskar_binary_read_int(h, group, OSKAR_SPLINES_TAG_NUM_KNOTS_X_THETA, index,
            &splines->num_knots_x_theta, status);
    oskar_binary_read_int(h, group, OSKAR_SPLINES_TAG_NUM_KNOTS_Y_PHI, index,
            &splines->num_knots_y_phi, status);
    oskar_binary_read_mem(h, oskar_splines_knots_x(splines),
            group, OSKAR_SPLINES_TAG_KNOTS_X_THETA, index, status);
    oskar_binary_read_mem(h, oskar_splines_knots_y(splines),
            group, OSKAR_SPLINES_TAG_KNOTS_Y_PHI, index, status);
    oskar_binary_read_mem(h, oskar_splines_coeff(splines),
            group, OSKAR_SPLINES_TAG_COEFF, index, status);
    oskar_binary_read_double(h, group, OSKAR_SPLINES_TAG_SMOOTHING_FACTOR,
            index, &splines->smoothing_factor, status);
}

#ifdef __cplusplus
}
#endif
