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

int oskar_image_read(oskar_Image* image, const char* filename, int idx)
{
    int err = 0, type;
    unsigned char grp = OSKAR_TAG_GROUP_IMAGE;
    FILE* stream;
    oskar_BinaryTagIndex* index = NULL;

    /* Sanity check on inputs. */
    if (filename == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Open the stream. */
    stream = fopen(filename, "rb");
    if (stream == NULL)
        return OSKAR_ERR_FILE_IO;

    /* Read the data type. */
    err = oskar_binary_stream_read_int(stream, &index, grp,
            OSKAR_IMAGE_TAG_DATA_TYPE, idx, &type);
    if (err) goto cleanup;

    /* Initialise the image. */
    err = oskar_image_init(image, type, OSKAR_LOCATION_CPU);
    if (err) goto cleanup;

    /* Read the dimensions. */
    err = oskar_binary_stream_read_int(stream, &index, grp,
            OSKAR_IMAGE_TAG_NUM_PIXELS_WIDTH, idx, &image->width);
    if (err) goto cleanup;
    err = oskar_binary_stream_read_int(stream, &index, grp,
            OSKAR_IMAGE_TAG_NUM_PIXELS_HEIGHT, idx, &image->height);
    if (err) goto cleanup;
    err = oskar_binary_stream_read_int(stream, &index, grp,
            OSKAR_IMAGE_TAG_NUM_POLS, idx, &image->num_pols);
    if (err) goto cleanup;
    err = oskar_binary_stream_read_int(stream, &index, grp,
            OSKAR_IMAGE_TAG_NUM_TIMES, idx, &image->num_times);
    if (err) goto cleanup;
    err = oskar_binary_stream_read_int(stream, &index, grp,
            OSKAR_IMAGE_TAG_NUM_CHANNELS, idx, &image->num_channels);
    if (err) goto cleanup;

    /* Read the dimension order. */
    err = oskar_mem_binary_stream_read(&image->dimension_order, stream, &index, grp,
            OSKAR_IMAGE_TAG_DIMENSION_ORDER, idx);
    if (err) goto cleanup;

    /* Read other image metadata. */
    err = oskar_binary_stream_read_int(stream, &index, grp,
            OSKAR_IMAGE_TAG_IMAGE_TYPE, idx, &image->image_type);
    if (err) goto cleanup;
    err = oskar_binary_stream_read_double(stream, &index, grp,
            OSKAR_IMAGE_TAG_CENTRE_RA, idx, &image->centre_ra_deg);
    if (err) goto cleanup;
    err = oskar_binary_stream_read_double(stream, &index, grp,
            OSKAR_IMAGE_TAG_CENTRE_DEC, idx, &image->centre_dec_deg);
    if (err) goto cleanup;
    err = oskar_binary_stream_read_double(stream, &index, grp,
            OSKAR_IMAGE_TAG_FOV_RA, idx, &image->fov_ra_deg);
    if (err) goto cleanup;
    err = oskar_binary_stream_read_double(stream, &index, grp,
            OSKAR_IMAGE_TAG_FOV_DEC, idx, &image->fov_dec_deg);
    if (err) goto cleanup;
    err = oskar_binary_stream_read_double(stream, &index, grp,
            OSKAR_IMAGE_TAG_TIME_START_MJD_UTC, idx, &image->time_start_mjd_utc);
    if (err) goto cleanup;
    err = oskar_binary_stream_read_double(stream, &index, grp,
            OSKAR_IMAGE_TAG_TIME_INC_SEC, idx, &image->time_inc_sec);
    if (err) goto cleanup;
    err = oskar_binary_stream_read_double(stream, &index, grp,
            OSKAR_IMAGE_TAG_FREQ_START_HZ, idx, &image->freq_start_hz);
    if (err) goto cleanup;
    err = oskar_binary_stream_read_double(stream, &index, grp,
            OSKAR_IMAGE_TAG_FREQ_INC_HZ, idx, &image->freq_inc_hz);
    if (err) goto cleanup;
    err = oskar_binary_stream_read_double(stream, &index, grp,
            OSKAR_IMAGE_TAG_MEAN, idx, &image->mean);
    if (err) goto cleanup;
    err = oskar_binary_stream_read_double(stream, &index, grp,
            OSKAR_IMAGE_TAG_VARIANCE, idx, &image->variance);
    if (err) goto cleanup;
    err = oskar_binary_stream_read_double(stream, &index, grp,
            OSKAR_IMAGE_TAG_MIN, idx, &image->min);
    if (err) goto cleanup;
    err = oskar_binary_stream_read_double(stream, &index, grp,
            OSKAR_IMAGE_TAG_MAX, idx, &image->max);
    if (err) goto cleanup;
    err = oskar_binary_stream_read_double(stream, &index, grp,
            OSKAR_IMAGE_TAG_RMS, idx, &image->rms);
    if (err) goto cleanup;

    /* Read the image data. */
    err = oskar_mem_binary_stream_read(&image->data, stream, &index,
            grp, OSKAR_IMAGE_TAG_IMAGE_DATA, idx);

    cleanup:
    oskar_binary_tag_index_free(&index);
    fclose(stream);
    return err;
}

#ifdef __cplusplus
}
#endif
