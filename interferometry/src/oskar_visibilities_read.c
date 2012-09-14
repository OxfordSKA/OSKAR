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
#include "interferometry/oskar_visibilities_init.h"
#include "interferometry/oskar_visibilities_read.h"
#include "utility/oskar_binary_file_read.h"
#include "utility/oskar_binary_tag_index_free.h"
#include "utility/oskar_BinaryTag.h"
#include "utility/oskar_mem_element_size.h"
#include "utility/oskar_mem_binary_file_read.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_visibilities_read(oskar_Visibilities* vis, const char* filename)
{
    /* Visibility metadata. */
    int num_channels = 0, num_times = 0, num_stations = 0;
    int amp_type = 0, err = 0;
    oskar_BinaryTagIndex* index = NULL;
    unsigned char grp = OSKAR_TAG_GROUP_VISIBILITY;

    /* Sanity check on inputs. */
    if (filename == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Read visibility dimensions. */
    err = oskar_binary_file_read_int(filename, &index, grp,
            OSKAR_VIS_TAG_NUM_CHANNELS, 0, &num_channels);
    if (err) goto cleanup;
    err = oskar_binary_file_read_int(filename, &index, grp,
            OSKAR_VIS_TAG_NUM_TIMES, 0, &num_times);
    if (err) goto cleanup;
    err = oskar_binary_file_read_int(filename, &index, grp,
            OSKAR_VIS_TAG_NUM_STATIONS, 0, &num_stations);
    if (err == OSKAR_ERR_BINARY_TAG_NOT_FOUND)
    {
        /* Check for number of baselines if number of stations not present. */
        int num_baselines = 0;
        err = oskar_binary_file_read_int(filename, &index, grp,
                OSKAR_VIS_TAG_NUM_BASELINES, 0, &num_baselines);
        if (err) goto cleanup;

        /* Convert baselines to stations (care using floating point here). */
        num_stations = (int) floor(0.5 +
                (1.0 + sqrt(1.0 + 8.0 * num_baselines)) / 2.0);
    }
    else if (err) goto cleanup;
    err = oskar_binary_file_read_int(filename, &index, grp,
            OSKAR_VIS_TAG_AMP_TYPE, 0, &amp_type);
    if (err) goto cleanup;

    /* Create the visibility structure. */
    oskar_visibilities_init(vis, amp_type, OSKAR_LOCATION_CPU,
            num_channels, num_times, num_stations, &err);
    if (err) return err;

    /* Optionally read the settings path (ignore the error code). */
    oskar_mem_binary_file_read(&vis->settings_path, filename, &index,
            OSKAR_TAG_GROUP_SETTINGS, OSKAR_TAG_SETTINGS_PATH, 0);

    /* Read visibility metadata. */
    err = oskar_binary_file_read_double(filename, &index, grp,
            OSKAR_VIS_TAG_FREQ_START_HZ, 0, &vis->freq_start_hz);
    if (err) goto cleanup;
    err = oskar_binary_file_read_double(filename, &index, grp,
            OSKAR_VIS_TAG_FREQ_INC_HZ, 0, &vis->freq_inc_hz);
    if (err) goto cleanup;
    err = oskar_binary_file_read_double(filename, &index, grp,
            OSKAR_VIS_TAG_TIME_START_MJD_UTC, 0, &vis->time_start_mjd_utc);
    if (err) goto cleanup;
    err = oskar_binary_file_read_double(filename, &index, grp,
            OSKAR_VIS_TAG_TIME_INC_SEC, 0, &vis->time_inc_seconds);
    if (err) goto cleanup;
    err = oskar_binary_file_read_double(filename, &index, grp,
            OSKAR_VIS_TAG_PHASE_CENTRE_RA, 0, &vis->phase_centre_ra_deg);
    if (err) goto cleanup;
    err = oskar_binary_file_read_double(filename, &index, grp,
            OSKAR_VIS_TAG_PHASE_CENTRE_DEC, 0, &vis->phase_centre_dec_deg);
    if (err) goto cleanup;

    /* Read the baseline coordinate arrays. */
    err = oskar_mem_binary_file_read(&vis->uu_metres, filename, &index, grp,
            OSKAR_VIS_TAG_BASELINE_UU, 0);
    if (err) goto cleanup;
    err = oskar_mem_binary_file_read(&vis->vv_metres, filename, &index, grp,
            OSKAR_VIS_TAG_BASELINE_VV, 0);
    if (err) goto cleanup;
    err = oskar_mem_binary_file_read(&vis->ww_metres, filename, &index, grp,
            OSKAR_VIS_TAG_BASELINE_WW, 0);
    if (err) goto cleanup;

    /* Read the visibility data. */
    err = oskar_mem_binary_file_read(&vis->amplitude, filename, &index, grp,
            OSKAR_VIS_TAG_AMPLITUDE, 0);
    if (err) goto cleanup;

    /* Try to read station coordinates, but don't worry if they're not here. */
    oskar_mem_binary_file_read(&vis->x_metres, filename, &index, grp,
            OSKAR_VIS_TAG_STATION_X, 0);
    oskar_mem_binary_file_read(&vis->y_metres, filename, &index, grp,
            OSKAR_VIS_TAG_STATION_Y, 0);
    oskar_mem_binary_file_read(&vis->z_metres, filename, &index, grp,
            OSKAR_VIS_TAG_STATION_Z, 0);

    cleanup:
    oskar_binary_tag_index_free(&index);
    return err;
}

#ifdef __cplusplus
}
#endif
