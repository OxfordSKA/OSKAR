/*
 * Copyright (c) 2011, The University of Oxford
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
#include "utility/oskar_binary_file_write.h"
#include "utility/oskar_BinaryTag.h"
#include "utility/oskar_file_exists.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_element_size.h"
#include "utility/oskar_mem_binary_file_read_raw.h"
#include "utility/oskar_mem_binary_file_write.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_visibilities_write(const char* filename, const oskar_Visibilities* vis)
{
    oskar_Mem temp;
    int err = 0, num_amps, num_coords;
    int uu_elements, vv_elements, ww_elements, amp_elements;
    int coord_type, amp_type, *dim;
    unsigned char grp = OSKAR_TAG_GROUP_VISIBILITY;

    /* Get the metadata. */
#ifdef __cplusplus
    uu_elements = vis->uu_metres.num_elements();
    vv_elements = vis->vv_metres.num_elements();
    ww_elements = vis->ww_metres.num_elements();
    amp_elements = vis->amplitude.num_elements();
    amp_type = vis->amplitude.type();
    coord_type = vis->uu_metres.type();
#else
    uu_elements = vis->uu_metres.private_num_elements;
    vv_elements = vis->vv_metres.private_num_elements;
    ww_elements = vis->ww_metres.private_num_elements;
    amp_elements = vis->amplitude.private_num_elements;
    amp_type = vis->amplitude.private_type;
    coord_type = vis->uu_metres.private_type;
#endif

    /* Sanity check on inputs. */
    if (filename == NULL || vis == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Check dimensions. */
    num_amps = vis->num_channels * vis->num_times * vis->num_baselines;
    num_coords = vis->num_times * vis->num_baselines;
    if (num_amps != amp_elements || num_coords != uu_elements ||
            num_coords != vv_elements || num_coords != ww_elements)
        return OSKAR_ERR_DIMENSION_MISMATCH;

    /* If the file already exists, remove it. */
    if (oskar_file_exists(filename))
        remove(filename);

    /* If settings path is set, write out the data. */
    if (vis->settings_path.data)
    {
        /* Write the settings path. */
        err = oskar_mem_binary_file_write(&vis->settings_path, filename,
                OSKAR_TAG_GROUP_SETTINGS, OSKAR_TAG_SETTINGS_PATH, 0, 0);
        if (err) return err;

        /* Write the settings file. */
        oskar_mem_init(&temp, OSKAR_CHAR, OSKAR_LOCATION_CPU, 0, 1);
        err = oskar_mem_binary_file_read_raw(&temp,
                (const char*) vis->settings_path.data);
        if (err) return err;
        err = oskar_mem_binary_file_write(&temp, filename,
                OSKAR_TAG_GROUP_SETTINGS, OSKAR_TAG_SETTINGS, 0, 0);
        oskar_mem_free(&temp);
        if (err) return err;
    }

    /* Write dimensions. */
    oskar_binary_file_write_int(filename, grp,
            OSKAR_VIS_TAG_NUM_CHANNELS, 0, vis->num_channels);
    oskar_binary_file_write_int(filename, grp,
            OSKAR_VIS_TAG_NUM_TIMES, 0, vis->num_times);
    oskar_binary_file_write_int(filename, grp,
            OSKAR_VIS_TAG_NUM_BASELINES, 0, vis->num_baselines);

    /* Write the dimension order. */
    oskar_mem_init(&temp, OSKAR_INT, OSKAR_LOCATION_CPU, 4, 1);
    dim = (int*) temp.data;
    dim[0] = OSKAR_VIS_DIM_CHANNEL;
    dim[1] = OSKAR_VIS_DIM_TIME;
    dim[2] = OSKAR_VIS_DIM_BASELINE;
    dim[3] = OSKAR_VIS_DIM_POLARISATION;
    oskar_mem_binary_file_write(&temp, filename, grp,
            OSKAR_VIS_TAG_DIMENSION_ORDER, 0, 0);
    oskar_mem_free(&temp);

    /* Write other visibility metadata. */
    oskar_binary_file_write_int(filename, grp,
            OSKAR_VIS_TAG_COORD_TYPE, 0, coord_type);
    oskar_binary_file_write_int(filename, grp,
            OSKAR_VIS_TAG_AMP_TYPE, 0, amp_type);
    oskar_binary_file_write_double(filename, grp,
            OSKAR_VIS_TAG_FREQ_START_HZ, 0, vis->freq_start_hz);
    oskar_binary_file_write_double(filename, grp,
            OSKAR_VIS_TAG_FREQ_INC_HZ, 0, vis->freq_inc_hz);
    oskar_binary_file_write_double(filename, grp,
            OSKAR_VIS_TAG_TIME_START_MJD_UTC, 0, vis->time_start_mjd_utc);
    oskar_binary_file_write_double(filename, grp,
            OSKAR_VIS_TAG_TIME_INC_SEC, 0, vis->time_inc_seconds);
    oskar_binary_file_write_int(filename, grp,
            OSKAR_VIS_TAG_POL_TYPE, 0, OSKAR_VIS_POL_TYPE_LINEAR);
    oskar_binary_file_write_int(filename, grp,
            OSKAR_VIS_TAG_BASELINE_COORD_UNIT, 0,
            OSKAR_VIS_BASELINE_COORD_UNIT_METRES);

    /* Write the baseline coordinate arrays. */
    oskar_mem_binary_file_write(&vis->uu_metres, filename,
            grp, OSKAR_VIS_TAG_BASELINE_UU, 0, 0);
    oskar_mem_binary_file_write(&vis->vv_metres, filename,
            grp, OSKAR_VIS_TAG_BASELINE_VV, 0, 0);
    oskar_mem_binary_file_write(&vis->ww_metres, filename,
            grp, OSKAR_VIS_TAG_BASELINE_WW, 0, 0);

    /* Write the visibility data. */
    err = oskar_mem_binary_file_write(&vis->amplitude, filename,
            grp, OSKAR_VIS_TAG_AMPLITUDE, 0, 0);

    return err;
}

#ifdef __cplusplus
}
#endif
