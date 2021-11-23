/*
 * Copyright (c) 2017-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/private_imager.h"
#include "imager/private_imager_read_dims.h"
#include "imager/oskar_imager.h"
#include "binary/oskar_binary.h"
#include "ms/oskar_measurement_set.h"
#include "vis/oskar_vis_header.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_read_dims_ms(oskar_Imager* h, const char* filename,
        int* status)
{
#ifndef OSKAR_NO_MS
    oskar_MeasurementSet* ms = 0;
    if (*status) return;

    /* Read the header. */
    oskar_log_message(h->log, 'M', 0, "Opening Measurement Set '%s'", filename);
    ms = oskar_ms_open_readonly(filename);
    if (!ms)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Set visibility meta-data. */
    oskar_imager_set_vis_frequency(h,
            oskar_ms_freq_start_hz(ms),
            oskar_ms_freq_inc_hz(ms),
            (int) oskar_ms_num_channels(ms));
    oskar_ms_close(ms);
#else
    (void) filename;
    oskar_log_error(h->log,
            "OSKAR was compiled without Measurement Set support.");
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
#endif
}


void oskar_imager_read_dims_vis(oskar_Imager* h, const char* filename,
        int* status)
{
    oskar_Binary* vis_file = 0;
    oskar_VisHeader* header = 0;
    if (*status) return;

    /* Read the header. */
    oskar_log_message(h->log, 'M', 0, "Opening '%s'", filename);
    vis_file = oskar_binary_create(filename, 'r', status);
    header = oskar_vis_header_read(vis_file, status);
    if (*status)
    {
        oskar_vis_header_free(header, status);
        oskar_binary_free(vis_file);
        return;
    }

    /* Set visibility meta-data. */
    oskar_imager_set_vis_frequency(h,
            oskar_vis_header_freq_start_hz(header),
            oskar_vis_header_freq_inc_hz(header),
            oskar_vis_header_num_channels_total(header));
    oskar_vis_header_free(header, status);
    oskar_binary_free(vis_file);
}


#ifdef __cplusplus
}
#endif
