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
#include "imaging/oskar_Image.h"
#include "utility/oskar_mem_init.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_image_init(oskar_Image* image, int type, int location)
{
    int err, *dim;

    /* Sanity check on inputs. */
    if (image == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Initialise memory. */
    err = oskar_mem_init(&image->data, type, location, 0, 1);
    if (err) return err;
    err = oskar_mem_init(&image->dimension_order, OSKAR_INT,
            OSKAR_LOCATION_CPU, 5, 1);
    err = oskar_mem_init(&image->settings_path, OSKAR_CHAR,
            OSKAR_LOCATION_CPU, 0, 1);
    if (err) return err;

    /* Set default dimension order. */
    dim = (int*) image->dimension_order.data;
    dim[0] = OSKAR_IMAGE_DIM_RA;
    dim[1] = OSKAR_IMAGE_DIM_DEC;
    dim[2] = OSKAR_IMAGE_DIM_POL;
    dim[3] = OSKAR_IMAGE_DIM_TIME;
    dim[4] = OSKAR_IMAGE_DIM_CHANNEL;

    /* Initialise meta-data. */
    image->image_type = OSKAR_IMAGE_TYPE_STOKES;
    image->centre_dec_deg = 0.0;
    image->centre_ra_deg = 0.0;
    image->fov_dec_deg = 0.0;
    image->fov_ra_deg = 0.0;
    image->freq_inc_hz = 0.0;
    image->freq_start_hz = 0.0;
    image->height = 0;
    image->max = 0.0;
    image->mean = 0.0;
    image->min = 0.0;
    image->num_channels = 0;
    image->num_pols = 0;
    image->num_times = 0;
    image->rms = 0.0;
    image->time_inc_sec = 0.0;
    image->time_start_mjd_utc = 0.0;
    image->variance = 0.0;
    image->width = 0;

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
