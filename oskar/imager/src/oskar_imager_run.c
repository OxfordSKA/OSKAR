/*
 * Copyright (c) 2016-2017, The University of Oxford
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

#include "imager/private_imager.h"
#include "imager/private_imager_read_coords.h"
#include "imager/private_imager_read_data.h"
#include "imager/private_imager_read_dims.h"
#include "imager/oskar_imager.h"

#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static int oskar_imager_is_ms(const char* filename);

void oskar_imager_run(oskar_Imager* h,
        int num_output_images, oskar_Mem** output_images,
        int num_output_grids, oskar_Mem** output_grids, int* status)
{
    int i, num_files, percent_done = 0, percent_next = 10;
    const char* filename;
    if (*status) return;

    /* Check input file has been set. */
    num_files = h->num_files;
    if (!h->input_files || num_files == 0)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Clear imager cache. */
    oskar_imager_reset_cache(h, status);

    /* Read dimension sizes. */
    for (i = 0; i < num_files; ++i)
    {
        if (*status) break;
        filename = h->input_files[i];
        if (oskar_imager_is_ms(filename))
            oskar_imager_read_dims_ms(h, filename, status);
        else
            oskar_imager_read_dims_vis(h, filename, status);
        if (*status && h->log)
            oskar_log_error(h->log, "Error opening file '%s'", filename);
    }

    /* Check for errors. */
    if (*status)
    {
        oskar_imager_reset_cache(h, status);
        return;
    }

    if (h->log)
    {
        oskar_log_message(h->log, 'M', 0, "Using %d frequency channel(s)",
                h->num_sel_freqs);
        if (h->num_sel_freqs > 0)
            oskar_log_message(h->log, 'M', 1, "Range %.3f MHz to %.3f MHz",
                    h->sel_freqs[0] * 1e-6,
                    h->sel_freqs[h->num_sel_freqs - 1] * 1e-6);
    }

    /* Check data ranges. */
    if (h->num_sel_freqs == 0)
    {
        if (h->log)
            oskar_log_error(h->log, "No data selected.");
        *status = OSKAR_ERR_OUT_OF_RANGE;
        oskar_imager_reset_cache(h, status);
        return;
    }

    /* Read baseline coordinates and weights if required. */
    if (h->weighting == OSKAR_WEIGHTING_UNIFORM ||
            h->algorithm == OSKAR_ALGORITHM_WPROJ)
    {
        oskar_imager_set_coords_only(h, 1);
        if (h->log)
            oskar_log_section(h->log, 'M', "Reading coordinates...");

        /* Loop over input files. */
        for (i = 0; i < num_files; ++i)
        {
            /* Read coordinates and weights. */
            if (*status) break;
            filename = h->input_files[i];
            if (h->log)
                oskar_log_message(h->log, 'M', 0, "Opening '%s'", filename);
            if (oskar_imager_is_ms(filename))
                oskar_imager_read_coords_ms(h, filename, i, num_files,
                        &percent_done, &percent_next, status);
            else
                oskar_imager_read_coords_vis(h, filename, i, num_files,
                        &percent_done, &percent_next, status);
        }
        oskar_imager_set_coords_only(h, 0);
    }

    /* Check for errors. */
    if (*status)
    {
        oskar_imager_reset_cache(h, status);
        return;
    }

    /* Initialise the algorithm. */
    if (h->log)
        oskar_log_section(h->log, 'M', "Initialising algorithm...");
    oskar_imager_check_init(h, status);
    if (h->log && !*status)
    {
        oskar_log_message(h->log, 'M', 0, "Plane size is %d x %d.",
                oskar_imager_plane_size(h), oskar_imager_plane_size(h));
        if (h->algorithm == OSKAR_ALGORITHM_WPROJ)
        {
            oskar_log_message(h->log, 'M', 0,
                    "Baseline W values (wavelengths)");
            oskar_log_message(h->log, 'M', 1, "Min: %.12e", h->ww_min);
            oskar_log_message(h->log, 'M', 1, "Max: %.12e", h->ww_max);
            oskar_log_message(h->log, 'M', 1, "RMS: %.12e", h->ww_rms);
            oskar_log_message(h->log, 'M', 0, "Using %d W-planes.",
                    oskar_imager_num_w_planes(h));
        }
        oskar_log_section(h->log, 'M', "Reading visibility data...");
    }

    /* Loop over input files. */
    percent_done = 0; percent_next = 10;
    for (i = 0; i < num_files; ++i)
    {
        /* Read visibility data. */
        if (*status) break;
        filename = h->input_files[i];
        if (h->log)
            oskar_log_message(h->log, 'M', 0, "Opening '%s'", filename);
        if (oskar_imager_is_ms(filename))
            oskar_imager_read_data_ms(h, filename, i, num_files,
                    &percent_done, &percent_next, status);
        else
            oskar_imager_read_data_vis(h, filename, i, num_files,
                    &percent_done, &percent_next, status);
    }

    /* Check for errors. */
    if (*status)
    {
        oskar_imager_reset_cache(h, status);
        return;
    }

    if (h->log)
        oskar_log_section(h->log, 'M', "Finalising %d image plane(s)...",
                h->num_planes);
    oskar_imager_finalise(h, num_output_images, output_images,
            num_output_grids, output_grids, status);
}


int oskar_imager_is_ms(const char* filename)
{
    size_t len;
    len = strlen(filename);
    if (len == 0) return 0;
    return (len >= 3) && (
            !strcmp(&(filename[len-3]), ".MS") ||
            !strcmp(&(filename[len-3]), ".ms") ) ? 1 : 0;
}


#ifdef __cplusplus
}
#endif
