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

#include "imaging/oskar_image_write.h"
#include "imaging/oskar_Image.h"
#include "utility/oskar_BinaryTag.h"
#include "utility/oskar_binary_stream_write.h"
#include "utility/oskar_binary_stream_write_header.h"
#include "utility/oskar_binary_stream_write_metadata.h"
#include "utility/oskar_mem_binary_stream_write.h"
#include "utility/oskar_mem_binary_file_read_raw.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_init.h"
#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_image_write(const oskar_Image* image, const char* filename, int idx)
{
    int err = 0, num, num_elements, type;
    unsigned char grp = OSKAR_TAG_GROUP_IMAGE;
    FILE* stream;

    /* Get the metadata. */
    num_elements = image->data.num_elements;
    type = image->data.type;

    /* Sanity check on inputs. */
    if (filename == NULL || image == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Check dimensions. */
    num = image->num_channels * image->num_times * image->num_pols *
            image->width * image->height;
    if (num != num_elements)
        return OSKAR_ERR_DIMENSION_MISMATCH;

    /* Open the stream. */
    stream = fopen(filename, "wb");
    if (stream == NULL)
        return OSKAR_ERR_FILE_IO;

    /* Write the header and common metadata. */
    err = oskar_binary_stream_write_header(stream);
    if (err) goto cleanup;
    err = oskar_binary_stream_write_metadata(stream);
    if (err) goto cleanup;

    /* If settings path is set, write out the data. */
    if (image->settings_path.data)
    {
        oskar_Mem temp;

        /* Write the settings path. */
        err = oskar_mem_binary_stream_write(&image->settings_path, stream,
                OSKAR_TAG_GROUP_SETTINGS, OSKAR_TAG_SETTINGS_PATH, idx, 0);
        if (err) goto cleanup;

        /* Write the settings file. */
        oskar_mem_init(&temp, OSKAR_CHAR, OSKAR_LOCATION_CPU, 0, 1);
        err = oskar_mem_binary_file_read_raw(&temp,
                (const char*) image->settings_path.data);
        if (err) goto cleanup;
        err = oskar_mem_binary_stream_write(&temp, stream,
                OSKAR_TAG_GROUP_SETTINGS, OSKAR_TAG_SETTINGS, idx, 0);
        oskar_mem_free(&temp);
        if (err) goto cleanup;
    }

    /* Write dimensions. */
    oskar_binary_stream_write_int(stream, grp,
            OSKAR_IMAGE_TAG_NUM_PIXELS_WIDTH, idx, image->width);
    oskar_binary_stream_write_int(stream, grp,
            OSKAR_IMAGE_TAG_NUM_PIXELS_HEIGHT, idx, image->height);
    oskar_binary_stream_write_int(stream, grp,
            OSKAR_IMAGE_TAG_NUM_POLS, idx, image->num_pols);
    oskar_binary_stream_write_int(stream, grp,
            OSKAR_IMAGE_TAG_NUM_TIMES, idx, image->num_times);
    oskar_binary_stream_write_int(stream, grp,
            OSKAR_IMAGE_TAG_NUM_CHANNELS, idx, image->num_channels);

    /* Write the dimension order. */
    oskar_mem_binary_stream_write(&image->dimension_order, stream, grp,
            OSKAR_IMAGE_TAG_DIMENSION_ORDER, idx, 0);

    /* Write other image metadata. */
    oskar_binary_stream_write_int(stream, grp,
            OSKAR_IMAGE_TAG_IMAGE_TYPE, idx, image->image_type);
    oskar_binary_stream_write_int(stream, grp,
            OSKAR_IMAGE_TAG_DATA_TYPE, idx, type);
    oskar_binary_stream_write_double(stream, grp,
            OSKAR_IMAGE_TAG_CENTRE_RA, idx, image->centre_ra_deg);
    oskar_binary_stream_write_double(stream, grp,
            OSKAR_IMAGE_TAG_CENTRE_DEC, idx, image->centre_dec_deg);
    oskar_binary_stream_write_double(stream, grp,
            OSKAR_IMAGE_TAG_FOV_RA, idx, image->fov_ra_deg);
    oskar_binary_stream_write_double(stream, grp,
            OSKAR_IMAGE_TAG_FOV_DEC, idx, image->fov_dec_deg);
    oskar_binary_stream_write_double(stream, grp,
            OSKAR_IMAGE_TAG_TIME_START_MJD_UTC, idx, image->time_start_mjd_utc);
    oskar_binary_stream_write_double(stream, grp,
            OSKAR_IMAGE_TAG_TIME_INC_SEC, idx, image->time_inc_sec);
    oskar_binary_stream_write_double(stream, grp,
            OSKAR_IMAGE_TAG_FREQ_START_HZ, idx, image->freq_start_hz);
    oskar_binary_stream_write_double(stream, grp,
            OSKAR_IMAGE_TAG_FREQ_INC_HZ, idx, image->freq_inc_hz);
    oskar_binary_stream_write_double(stream, grp,
            OSKAR_IMAGE_TAG_MEAN, idx, image->mean);
    oskar_binary_stream_write_double(stream, grp,
            OSKAR_IMAGE_TAG_VARIANCE, idx, image->variance);
    oskar_binary_stream_write_double(stream, grp,
            OSKAR_IMAGE_TAG_MIN, idx, image->min);
    oskar_binary_stream_write_double(stream, grp,
            OSKAR_IMAGE_TAG_MAX, idx, image->max);
    oskar_binary_stream_write_double(stream, grp,
            OSKAR_IMAGE_TAG_RMS, idx, image->rms);

    /* Write the image data. */
    err = oskar_mem_binary_stream_write(&image->data, stream,
            grp, OSKAR_IMAGE_TAG_IMAGE_DATA, idx, 0);

    cleanup:
    fclose(stream);
    return err;
}

#ifdef __cplusplus
}
#endif
