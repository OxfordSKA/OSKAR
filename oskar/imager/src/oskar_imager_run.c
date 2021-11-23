/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/private_imager.h"
#include "imager/private_imager_read_coords.h"
#include "imager/private_imager_read_data.h"
#include "imager/private_imager_read_dims.h"
#include "imager/oskar_imager.h"
#include "utility/oskar_get_error_string.h"

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
    const char* filename = 0;
    int i = 0, num_files = 0, percent_done = 0, percent_next = 10;
    if (*status || !h) return;
    oskar_log_section(h->log, 'M', "Starting imager...");

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
        {
            oskar_imager_read_dims_ms(h, filename, status);
        }
        else
        {
            oskar_imager_read_dims_vis(h, filename, status);
        }
        if (*status)
        {
            oskar_log_error(h->log, "Error opening file '%s'", filename);
        }
    }

    /* Check for errors. */
    if (*status)
    {
        oskar_imager_reset_cache(h, status);
        return;
    }

    /* Check data ranges. */
    if (h->num_sel_freqs == 0)
    {
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
        oskar_log_section(h->log, 'M', "Reading coordinates...");

        /* Loop over input files. */
        for (i = 0; i < num_files; ++i)
        {
            /* Read coordinates and weights. */
            if (*status) break;
            filename = h->input_files[i];
            if (oskar_imager_is_ms(filename))
            {
                oskar_imager_read_coords_ms(h, filename, i, num_files,
                        &percent_done, &percent_next, status);
            }
            else
            {
                oskar_imager_read_coords_vis(h, filename, i, num_files,
                        &percent_done, &percent_next, status);
            }
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
    oskar_imager_check_init(h, status);
    if (!*status)
    {
        oskar_log_section(h->log, 'M', "Reading visibility data...");
    }

    /* Loop over input files. */
    percent_done = 0; percent_next = 10;
    for (i = 0; i < num_files; ++i)
    {
        /* Read visibility data. */
        if (*status) break;
        filename = h->input_files[i];
        if (oskar_imager_is_ms(filename))
        {
            oskar_imager_read_data_ms(h, filename, i, num_files,
                    &percent_done, &percent_next, status);
        }
        else
        {
            oskar_imager_read_data_vis(h, filename, i, num_files,
                    &percent_done, &percent_next, status);
        }
    }

    /* Check for errors. */
    if (*status)
    {
        oskar_imager_reset_cache(h, status);
        return;
    }

    /* Finalise. */
    oskar_imager_finalise(h, num_output_images, output_images,
            num_output_grids, output_grids, status);
}


int oskar_imager_is_ms(const char* filename)
{
    size_t len = 0;
    len = strlen(filename);
    if (len == 0) return 0;
    return (len >= 3) && (
            !strcmp(&(filename[len-3]), ".MS") ||
            !strcmp(&(filename[len-3]), ".ms") ) ? 1 : 0;
}


#ifdef __cplusplus
}
#endif
