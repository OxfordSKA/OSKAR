/*
 * Copyright (c) 2012, The University of Oxford
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

#include "oskar_global.h"
#include "interferometry/oskar_Visibilities.h"
#include "interferometry/oskar_visibilities_write.h"
#include "utility/oskar_binary_stream_write.h"
#include "utility/oskar_binary_stream_write_header.h"
#include "utility/oskar_binary_stream_write_metadata.h"
#include "utility/oskar_BinaryTag.h"
#include "utility/oskar_file_exists.h"
#include "utility/oskar_log_file_data.h"
#include "utility/oskar_log_message.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_element_size.h"
#include "utility/oskar_mem_binary_stream_write.h"
#include "utility/oskar_mem_binary_file_read_raw.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_visibilities_write(const oskar_Visibilities* vis, oskar_Log* log,
        const char* filename, int* status)
{
    oskar_Mem temp;
    int num_amps, num_coords;
    int uu_elements, vv_elements, ww_elements, amp_elements;
    int coord_type, amp_type, *dim;
    unsigned char grp = OSKAR_TAG_GROUP_VISIBILITY;
    FILE* stream;
    char* log_data = 0;
    long log_size = 0;

    /* Check all inputs. */
    if (!vis || !filename || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the metadata. */
    uu_elements = vis->uu_metres.num_elements;
    vv_elements = vis->vv_metres.num_elements;
    ww_elements = vis->ww_metres.num_elements;
    amp_elements = vis->amplitude.num_elements;
    amp_type = vis->amplitude.type;
    coord_type = vis->uu_metres.type;

    /* Check dimensions. */
    num_amps = vis->num_channels * vis->num_times * vis->num_baselines;
    num_coords = vis->num_times * vis->num_baselines;
    if (num_amps != amp_elements || num_coords != uu_elements ||
            num_coords != vv_elements || num_coords != ww_elements)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Open the stream. */
    stream = fopen(filename, "wb");
    if (stream == NULL)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Write a log message. */
    oskar_log_message(log, 0, "Writing OSKAR visibility file: '%s'", filename);

    /* Write the header and common metadata. */
    oskar_binary_stream_write_header(stream, status);
    oskar_binary_stream_write_metadata(stream, status);

    /* Check if safe to proceed. */
    if (*status) return;

    /* If settings path is set, write out the data. */
    if (vis->settings_path.data)
    {
        if (strlen(vis->settings_path.data) > 0)
        {
            /* Write the settings path. */
            oskar_mem_binary_stream_write(&vis->settings_path, stream,
                    OSKAR_TAG_GROUP_SETTINGS, OSKAR_TAG_SETTINGS_PATH, 0, 0,
                    status);

            /* Check the file exists */
            if (oskar_file_exists((const char*)vis->settings_path.data))
            {
                /* Write the settings file. */
                oskar_mem_init(&temp, OSKAR_CHAR, OSKAR_LOCATION_CPU, 0, 1, status);
                oskar_mem_binary_file_read_raw(&temp,
                        (const char*) vis->settings_path.data, status);
                oskar_mem_binary_stream_write(&temp, stream,
                        OSKAR_TAG_GROUP_SETTINGS, OSKAR_TAG_SETTINGS, 0, 0, status);
                oskar_mem_free(&temp, status);
            }
        }
    }
    /* If log exists, then write it out. */
    log_data = oskar_log_file_data(log, &log_size);
    if (log_data)
    {
        oskar_binary_stream_write(stream, OSKAR_CHAR,
                OSKAR_TAG_GROUP_RUN, OSKAR_TAG_RUN_LOG, 0, log_size, log_data,
                status);
        free(log_data);
    }

    /* Write dimensions. */
    oskar_binary_stream_write_int(stream, grp,
            OSKAR_VIS_TAG_NUM_CHANNELS, 0, vis->num_channels, status);
    oskar_binary_stream_write_int(stream, grp,
            OSKAR_VIS_TAG_NUM_TIMES, 0, vis->num_times, status);
    oskar_binary_stream_write_int(stream, grp,
            OSKAR_VIS_TAG_NUM_STATIONS, 0, vis->num_stations, status);
    oskar_binary_stream_write_int(stream, grp,
            OSKAR_VIS_TAG_NUM_BASELINES, 0, vis->num_baselines, status);

    /* Write the dimension order. */
    if (*status) return;
    oskar_mem_init(&temp, OSKAR_INT, OSKAR_LOCATION_CPU, 4, 1, status);
    dim = (int*) temp.data;
    dim[0] = OSKAR_VIS_DIM_CHANNEL;
    dim[1] = OSKAR_VIS_DIM_TIME;
    dim[2] = OSKAR_VIS_DIM_BASELINE;
    dim[3] = OSKAR_VIS_DIM_POLARISATION;
    oskar_mem_binary_stream_write(&temp, stream, grp,
            OSKAR_VIS_TAG_DIMENSION_ORDER, 0, 0, status);
    oskar_mem_free(&temp, status);

    /* Write other visibility metadata. */
    oskar_binary_stream_write_int(stream, grp,
            OSKAR_VIS_TAG_COORD_TYPE, 0, coord_type, status);
    oskar_binary_stream_write_int(stream, grp,
            OSKAR_VIS_TAG_AMP_TYPE, 0, amp_type, status);
    oskar_binary_stream_write_double(stream, grp,
            OSKAR_VIS_TAG_FREQ_START_HZ, 0, vis->freq_start_hz, status);
    oskar_binary_stream_write_double(stream, grp,
            OSKAR_VIS_TAG_FREQ_INC_HZ, 0, vis->freq_inc_hz, status);
    oskar_binary_stream_write_double(stream, grp,
            OSKAR_VIS_TAG_TIME_START_MJD_UTC, 0, vis->time_start_mjd_utc,
            status);
    oskar_binary_stream_write_double(stream, grp,
            OSKAR_VIS_TAG_TIME_INC_SEC, 0, vis->time_inc_seconds, status);
    oskar_binary_stream_write_int(stream, grp,
            OSKAR_VIS_TAG_POL_TYPE, 0, OSKAR_VIS_POL_TYPE_LINEAR, status);
    oskar_binary_stream_write_int(stream, grp,
            OSKAR_VIS_TAG_BASELINE_COORD_UNIT, 0,
            OSKAR_VIS_BASELINE_COORD_UNIT_METRES, status);
    oskar_binary_stream_write_int(stream, grp,
            OSKAR_VIS_TAG_STATION_COORD_UNIT, 0,
            OSKAR_VIS_STATION_COORD_UNIT_METRES, status);
    oskar_binary_stream_write_double(stream, grp,
            OSKAR_VIS_TAG_PHASE_CENTRE_RA, 0, vis->phase_centre_ra_deg, status);
    oskar_binary_stream_write_double(stream, grp,
            OSKAR_VIS_TAG_PHASE_CENTRE_DEC, 0, vis->phase_centre_dec_deg,
            status);

    /* Write the baseline coordinate arrays. */
    oskar_mem_binary_stream_write(&vis->uu_metres, stream,
            grp, OSKAR_VIS_TAG_BASELINE_UU, 0, 0, status);
    oskar_mem_binary_stream_write(&vis->vv_metres, stream,
            grp, OSKAR_VIS_TAG_BASELINE_VV, 0, 0, status);
    oskar_mem_binary_stream_write(&vis->ww_metres, stream,
            grp, OSKAR_VIS_TAG_BASELINE_WW, 0, 0, status);

    /* Write the visibility data. */
    oskar_mem_binary_stream_write(&vis->amplitude, stream,
            grp, OSKAR_VIS_TAG_AMPLITUDE, 0, 0, status);

    /* Write the station coordinate arrays. */
    oskar_mem_binary_stream_write(&vis->x_metres, stream,
            grp, OSKAR_VIS_TAG_STATION_X, 0, 0, status);
    oskar_mem_binary_stream_write(&vis->y_metres, stream,
            grp, OSKAR_VIS_TAG_STATION_Y, 0, 0, status);
    oskar_mem_binary_stream_write(&vis->z_metres, stream,
            grp, OSKAR_VIS_TAG_STATION_Z, 0, 0, status);

    /* Close the file. */
    fclose(stream);
}

#ifdef __cplusplus
}
#endif
