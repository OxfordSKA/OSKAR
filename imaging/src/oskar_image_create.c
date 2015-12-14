/*
 * Copyright (c) 2014-2015, The University of Oxford
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

#ifdef __cplusplus
extern "C" {
#endif

oskar_Image* oskar_image_create(int type, int location, int* status)
{
    oskar_Image* image = 0;

    /* Allocate the structure. */
    image = (oskar_Image*) malloc(sizeof(oskar_Image));

    /* Initialise memory. */
    image->data = oskar_mem_create(type, location, 0, status);
    image->settings_path = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 0,
            status);

    /* Set default dimension order. */
    image->dimension_order[0] = OSKAR_IMAGE_DIM_LONGITUDE;
    image->dimension_order[1] = OSKAR_IMAGE_DIM_LATITUDE;
    image->dimension_order[2] = OSKAR_IMAGE_DIM_POL;
    image->dimension_order[3] = OSKAR_IMAGE_DIM_TIME;
    image->dimension_order[4] = OSKAR_IMAGE_DIM_CHANNEL;

    /* Initialise meta-data. */
    image->grid_type = OSKAR_IMAGE_GRID_TYPE_RECTILINEAR;
    image->coord_frame = OSKAR_IMAGE_COORD_FRAME_UNDEF;
    image->image_type = OSKAR_IMAGE_TYPE_UNDEF;
    image->centre_lat_deg = 0.0;
    image->centre_lon_deg = 0.0;
    image->fov_lat_deg = 0.0;
    image->fov_lon_deg = 0.0;
    image->freq_inc_hz = 0.0;
    image->freq_start_hz = 0.0;
    image->height = 0;
    image->num_channels = 0;
    image->num_pols = 0;
    image->num_times = 0;
    image->time_inc_sec = 0.0;
    image->time_start_mjd_utc = 0.0;
    image->width = 0;

    /* Return a handle to the image. */
    return image;
}

#ifdef __cplusplus
}
#endif
