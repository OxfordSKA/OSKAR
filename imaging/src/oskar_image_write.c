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

#include "imaging/oskar_image_write.h"
#include "imaging/oskar_Image.h"
#include "utility/oskar_BinaryTag.h"
#include "utility/oskar_binary_stream_write.h"
#include "utility/oskar_binary_stream_write_header.h"
#include "utility/oskar_binary_stream_write_metadata.h"
#include "utility/oskar_log_file_data.h"
#include "utility/oskar_log_message.h"
#include "utility/oskar_mem_binary_stream_write.h"
#include "utility/oskar_mem_binary_file_read_raw.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_init.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_image_write(const oskar_Image* image, oskar_Log* log,
        const char* filename, int idx, int* status)
{
    int num, type;
    unsigned char grp = OSKAR_TAG_GROUP_IMAGE;
    FILE* stream;
    char* log_data = 0;
    long log_size = 0;

    /* Check all inputs. */
    if (!image || !filename || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the metadata. */
    type = image->data.type;

    /* Check dimensions. */
    num = image->num_channels * image->num_times * image->num_pols *
            image->width * image->height;
    if (num != image->data.num_elements)
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
    oskar_log_message(log, 0, "Writing OSKAR image file: '%s'", filename);

    /* Write the header and common metadata. */
    oskar_binary_stream_write_header(stream, status);
    oskar_binary_stream_write_metadata(stream, status);

    /* If settings path is set, write out the data. */
    if (image->settings_path.data)
    {
        if (strlen(image->settings_path.data) > 0)
        {
            oskar_Mem temp;
            /* Write the settings path. */
            oskar_mem_binary_stream_write(&image->settings_path, stream,
                    OSKAR_TAG_GROUP_SETTINGS, OSKAR_TAG_SETTINGS_PATH, idx, 0,
                    status);

            /* Check the file exists */
            if (oskar_file_exists((const char*)image->settings_path.data))
            {
                /* Write the settings file. */
                oskar_mem_init(&temp, OSKAR_CHAR, OSKAR_LOCATION_CPU, 0, 1, status);
                oskar_mem_binary_file_read_raw(&temp,
                        (const char*) image->settings_path.data, status);
                oskar_mem_binary_stream_write(&temp, stream,
                        OSKAR_TAG_GROUP_SETTINGS, OSKAR_TAG_SETTINGS, idx, 0, status);
                oskar_mem_free(&temp, status);
            }
        }
    }

    /* If log exists, then write it out. */
    log_data = oskar_log_file_data(log, &log_size);
    if (log_data)
    {
        oskar_binary_stream_write(stream, OSKAR_CHAR,
                OSKAR_TAG_GROUP_RUN, OSKAR_TAG_RUN_LOG, idx, log_size,
                log_data, status);
        free(log_data);
    }

    /* Write dimensions. */
    oskar_binary_stream_write_int(stream, grp,
            OSKAR_IMAGE_TAG_NUM_PIXELS_WIDTH, idx, image->width, status);
    oskar_binary_stream_write_int(stream, grp,
            OSKAR_IMAGE_TAG_NUM_PIXELS_HEIGHT, idx, image->height, status);
    oskar_binary_stream_write_int(stream, grp,
            OSKAR_IMAGE_TAG_NUM_POLS, idx, image->num_pols, status);
    oskar_binary_stream_write_int(stream, grp,
            OSKAR_IMAGE_TAG_NUM_TIMES, idx, image->num_times, status);
    oskar_binary_stream_write_int(stream, grp,
            OSKAR_IMAGE_TAG_NUM_CHANNELS, idx, image->num_channels, status);

    /* Write the dimension order. */
    oskar_binary_stream_write(stream, OSKAR_INT, grp,
            OSKAR_IMAGE_TAG_DIMENSION_ORDER, idx,
            sizeof(image->dimension_order), image->dimension_order, status);

    /* Write other image metadata. */
    oskar_binary_stream_write_int(stream, grp,
            OSKAR_IMAGE_TAG_IMAGE_TYPE, idx, image->image_type, status);
    oskar_binary_stream_write_int(stream, grp,
            OSKAR_IMAGE_TAG_DATA_TYPE, idx, type, status);
    oskar_binary_stream_write_double(stream, grp,
            OSKAR_IMAGE_TAG_CENTRE_RA, idx, image->centre_ra_deg, status);
    oskar_binary_stream_write_double(stream, grp,
            OSKAR_IMAGE_TAG_CENTRE_DEC, idx, image->centre_dec_deg, status);
    oskar_binary_stream_write_double(stream, grp,
            OSKAR_IMAGE_TAG_FOV_RA, idx, image->fov_ra_deg, status);
    oskar_binary_stream_write_double(stream, grp,
            OSKAR_IMAGE_TAG_FOV_DEC, idx, image->fov_dec_deg, status);
    oskar_binary_stream_write_double(stream, grp,
            OSKAR_IMAGE_TAG_TIME_START_MJD_UTC, idx, image->time_start_mjd_utc,
            status);
    oskar_binary_stream_write_double(stream, grp,
            OSKAR_IMAGE_TAG_TIME_INC_SEC, idx, image->time_inc_sec, status);
    oskar_binary_stream_write_double(stream, grp,
            OSKAR_IMAGE_TAG_FREQ_START_HZ, idx, image->freq_start_hz, status);
    oskar_binary_stream_write_double(stream, grp,
            OSKAR_IMAGE_TAG_FREQ_INC_HZ, idx, image->freq_inc_hz, status);

    /* Write the image data. */
    oskar_mem_binary_stream_write(&image->data, stream,
            grp, OSKAR_IMAGE_TAG_IMAGE_DATA, idx, 0, status);

    /* Close the file. */
    fclose(stream);
}

#ifdef __cplusplus
}
#endif
