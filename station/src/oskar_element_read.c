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

#ifdef __cplusplus
extern "C" {
#endif

static void read_splines(FILE* fhan, oskar_Splines* splines,
        oskar_BinaryTagIndex** idx, int index, int* status);

void oskar_element_read(oskar_Element* data, int port, const char* filename,
        int* status)
{
    FILE* fhan = 0;
    oskar_Splines *theta_re, *theta_im, *phi_re, *phi_im;
    oskar_BinaryTagIndex* idx = 0;

    /* Check all inputs. */
    if (!data || !filename  || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get pointers based on port number. */
    if (port == 1)
    {
        theta_re = oskar_element_x_theta_re(data);
        theta_im = oskar_element_x_theta_im(data);
        phi_re   = oskar_element_x_phi_re(data);
        phi_im   = oskar_element_x_phi_im(data);
    }
    else if (port == 2)
    {
        theta_re = oskar_element_y_theta_re(data);
        theta_im = oskar_element_y_theta_im(data);
        phi_re   = oskar_element_y_phi_re(data);
        phi_im   = oskar_element_y_phi_im(data);
    }
    else
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    /* Read data from binary file. */
    fhan = fopen(filename, "rb");

    /* Read data for [theta_re], [theta_im], [phi_re], [phi_im] surfaces. */
    read_splines(fhan, theta_re, &idx, 0, status);
    read_splines(fhan, theta_im, &idx, 1, status);
    read_splines(fhan, phi_re, &idx, 2, status);
    read_splines(fhan, phi_im, &idx, 3, status);

    /* Close the file. */
    fclose(fhan);

    /* Free the tag index. */
    oskar_binary_tag_index_free(idx, status);
}

static void read_splines(FILE* fhan, oskar_Splines* splines,
        oskar_BinaryTagIndex** idx, int index, int* status)
{
    unsigned char group;
    group = OSKAR_TAG_GROUP_SPLINE_DATA;
    oskar_binary_stream_read_int(fhan, idx, group,
            OSKAR_SPLINES_TAG_NUM_KNOTS_X, index,
            &splines->num_knots_x, status);
    oskar_binary_stream_read_int(fhan, idx, group,
            OSKAR_SPLINES_TAG_NUM_KNOTS_Y, index,
            &splines->num_knots_y, status);
    oskar_mem_binary_stream_read(oskar_splines_knots_x(splines),
            fhan, idx, group, OSKAR_SPLINES_TAG_KNOTS_X, index, status);
    oskar_mem_binary_stream_read(oskar_splines_knots_y(splines),
            fhan, idx, group, OSKAR_SPLINES_TAG_KNOTS_Y, index, status);
    oskar_mem_binary_stream_read(oskar_splines_coeff(splines),
            fhan, idx, group, OSKAR_SPLINES_TAG_COEFF, index, status);
    oskar_binary_stream_read_double(fhan, idx, group,
            OSKAR_SPLINES_TAG_SMOOTHING_FACTOR, index,
            &splines->smoothing_factor, status);
}

#ifdef __cplusplus
}
#endif
