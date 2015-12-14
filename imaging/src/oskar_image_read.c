/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include <private_image.h>
#include <oskar_image.h>
#include <oskar_binary.h>
#include <oskar_binary_read_mem.h>
#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Image* oskar_image_read(const char* filename, int idx, int* status)
{
    int type, tag_error = 0;
    unsigned char grp = OSKAR_TAG_GROUP_IMAGE;
    oskar_Binary* h = 0;
    oskar_Image* image = 0;

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* Create the handle. */
    h = oskar_binary_create(filename, 'r', status);

    /* Read the data type. */
    oskar_binary_read_int(h, grp, OSKAR_IMAGE_TAG_DATA_TYPE, idx, &type,
            status);

    /* Check if safe to proceed. */
    if (*status)
    {
        oskar_binary_free(h);
        return 0;
    }

    /* Initialise the image. */
    image = oskar_image_create(type, OSKAR_CPU, status);

    /* Optionally read the settings path (ignore the error code). */
    oskar_binary_read_mem(h, image->settings_path,
            OSKAR_TAG_GROUP_SETTINGS, OSKAR_TAG_SETTINGS_PATH, 0, &tag_error);

    /* Read the dimensions. */
    oskar_binary_read_int(h, grp,
            OSKAR_IMAGE_TAG_NUM_PIXELS_WIDTH, idx, &image->width, status);
    oskar_binary_read_int(h, grp,
            OSKAR_IMAGE_TAG_NUM_PIXELS_HEIGHT, idx, &image->height, status);
    oskar_binary_read_int(h, grp,
            OSKAR_IMAGE_TAG_NUM_POLS, idx, &image->num_pols, status);
    oskar_binary_read_int(h, grp,
            OSKAR_IMAGE_TAG_NUM_TIMES, idx, &image->num_times, status);
    oskar_binary_read_int(h, grp,
            OSKAR_IMAGE_TAG_NUM_CHANNELS, idx, &image->num_channels, status);

    /* Read the dimension order. */
    oskar_binary_read(h, OSKAR_INT, grp, OSKAR_IMAGE_TAG_DIMENSION_ORDER, idx,
            sizeof(image->dimension_order), image->dimension_order, status);

    /* Read other image metadata. */
    oskar_binary_read_int(h, grp,
            OSKAR_IMAGE_TAG_IMAGE_TYPE, idx, &image->image_type, status);

    oskar_binary_read_double(h, grp,
            OSKAR_IMAGE_TAG_TIME_START_MJD_UTC, idx, &image->time_start_mjd_utc,
            status);
    oskar_binary_read_double(h, grp,
            OSKAR_IMAGE_TAG_TIME_INC_SEC, idx, &image->time_inc_sec, status);
    oskar_binary_read_double(h, grp,
            OSKAR_IMAGE_TAG_FREQ_START_HZ, idx, &image->freq_start_hz, status);
    oskar_binary_read_double(h, grp,
            OSKAR_IMAGE_TAG_FREQ_INC_HZ, idx, &image->freq_inc_hz, status);

    /* Optionally read pixel grid type and coordinate frame. These are optional
     * to maintain compatibility with OSKAR <= v2.3.1 */
    tag_error = 0;
    oskar_binary_read_int(h, grp,
            OSKAR_IMAGE_TAG_GRID_TYPE, idx, &image->grid_type, &tag_error);
    oskar_binary_read_int(h, grp,
            OSKAR_IMAGE_TAG_COORD_FRAME, idx, &image->coord_frame, &tag_error);

    /* Rectilinear grid type tags */
    tag_error = 0;
    oskar_binary_read_double(h, grp, OSKAR_IMAGE_TAG_CENTRE_LONGITUDE, idx,
            &image->centre_lon_deg, &tag_error);
    oskar_binary_read_double(h, grp, OSKAR_IMAGE_TAG_CENTRE_LATITUDE, idx,
            &image->centre_lat_deg, &tag_error);
    oskar_binary_read_double(h, grp, OSKAR_IMAGE_TAG_FOV_LONGITUDE, idx,
            &image->fov_lon_deg, &tag_error);
    oskar_binary_read_double(h, grp, OSKAR_IMAGE_TAG_FOV_LATITUDE, idx,
            &image->fov_lat_deg, &tag_error);

    /* HEALPix grid type tags */
    tag_error = 0;
    oskar_binary_read_int(h, grp, OSKAR_IMAGE_TAG_HEALPIX_NSIDE, idx,
            &image->healpix_nside, &tag_error);

    /* Read the image data. */
    oskar_binary_read_mem(h, image->data, grp, OSKAR_IMAGE_TAG_IMAGE_DATA, idx,
            status);

    /* Release the handle. */
    oskar_binary_free(h);

    /* Return a handle to the image. */
    return image;
}

#ifdef __cplusplus
}
#endif
