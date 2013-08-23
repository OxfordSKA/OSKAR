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

#include "imaging/oskar_image_init.h"
#include "imaging/oskar_image_read.h"
#include "imaging/oskar_Image.h"
#include "utility/oskar_BinaryTag.h"
#include "utility/oskar_binary_stream_read.h"
#include "utility/oskar_binary_tag_index_free.h"
#include "utility/oskar_mem_binary_stream_read.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_init.h"
#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_image_read(oskar_Image* image, const char* filename, int idx,
        int* status)
{
    int type, tag_error = 0;
    unsigned char grp = OSKAR_TAG_GROUP_IMAGE;
    FILE* stream;
    oskar_BinaryTagIndex* index = NULL;

    /* Check all inputs. */
    if (!image || !filename || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Open the stream. */
    stream = fopen(filename, "rb");
    if (stream == NULL)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Read the data type. */
    oskar_binary_stream_read_int(stream, &index, grp,
            OSKAR_IMAGE_TAG_DATA_TYPE, idx, &type, status);

    /* Check if safe to proceed. */
    if (*status)
    {
        oskar_binary_tag_index_free(index, status);
        fclose(stream);
        return;
    }

    /* Initialise the image. */
    oskar_image_init(image, type, OSKAR_LOCATION_CPU, status);

    /* Optionally read the settings path (ignore the error code). */
    oskar_mem_binary_stream_read(&image->settings_path, stream, &index,
            OSKAR_TAG_GROUP_SETTINGS, OSKAR_TAG_SETTINGS_PATH, 0, &tag_error);

    /* Read the dimensions. */
    oskar_binary_stream_read_int(stream, &index, grp,
            OSKAR_IMAGE_TAG_NUM_PIXELS_WIDTH, idx, &image->width, status);
    oskar_binary_stream_read_int(stream, &index, grp,
            OSKAR_IMAGE_TAG_NUM_PIXELS_HEIGHT, idx, &image->height, status);
    oskar_binary_stream_read_int(stream, &index, grp,
            OSKAR_IMAGE_TAG_NUM_POLS, idx, &image->num_pols, status);
    oskar_binary_stream_read_int(stream, &index, grp,
            OSKAR_IMAGE_TAG_NUM_TIMES, idx, &image->num_times, status);
    oskar_binary_stream_read_int(stream, &index, grp,
            OSKAR_IMAGE_TAG_NUM_CHANNELS, idx, &image->num_channels, status);

    /* Read the dimension order. */
    oskar_binary_stream_read(stream, &index, OSKAR_INT, grp,
            OSKAR_IMAGE_TAG_DIMENSION_ORDER, idx, sizeof(image->dimension_order),
            image->dimension_order, status);

    /* Read other image metadata. */
    oskar_binary_stream_read_int(stream, &index, grp,
            OSKAR_IMAGE_TAG_IMAGE_TYPE, idx, &image->image_type, status);
    oskar_binary_stream_read_double(stream, &index, grp,
            OSKAR_IMAGE_TAG_CENTRE_RA, idx, &image->centre_ra_deg, status);
    oskar_binary_stream_read_double(stream, &index, grp,
            OSKAR_IMAGE_TAG_CENTRE_DEC, idx, &image->centre_dec_deg, status);
    oskar_binary_stream_read_double(stream, &index, grp,
            OSKAR_IMAGE_TAG_FOV_RA, idx, &image->fov_ra_deg, status);
    oskar_binary_stream_read_double(stream, &index, grp,
            OSKAR_IMAGE_TAG_FOV_DEC, idx, &image->fov_dec_deg, status);
    oskar_binary_stream_read_double(stream, &index, grp,
            OSKAR_IMAGE_TAG_TIME_START_MJD_UTC, idx, &image->time_start_mjd_utc,
            status);
    oskar_binary_stream_read_double(stream, &index, grp,
            OSKAR_IMAGE_TAG_TIME_INC_SEC, idx, &image->time_inc_sec, status);
    oskar_binary_stream_read_double(stream, &index, grp,
            OSKAR_IMAGE_TAG_FREQ_START_HZ, idx, &image->freq_start_hz, status);
    oskar_binary_stream_read_double(stream, &index, grp,
            OSKAR_IMAGE_TAG_FREQ_INC_HZ, idx, &image->freq_inc_hz, status);

    /* Read the image data. */
    oskar_mem_binary_stream_read(&image->data, stream, &index,
            grp, OSKAR_IMAGE_TAG_IMAGE_DATA, idx, status);

    /* Free the index and close the stream. */
    oskar_binary_tag_index_free(index, status);
    fclose(stream);
}

#ifdef __cplusplus
}
#endif
