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

#ifndef OSKAR_IMAGE_H_
#define OSKAR_IMAGE_H_

#include "utility/oskar_Mem.h"

/**
 * @struct oskar_Image
 *
 * @brief Structure to hold image (cube) data.
 *
 * @details
 * This structure holds data for a multi-dimensional image hyper-cube.
 */
struct oskar_Image
{
    oskar_Mem data;
    oskar_Mem dimension_order;
    oskar_Mem settings_path;
    int width;
    int height;
    int num_pols;
    int num_times;
    int num_channels;
    double centre_ra_deg;
    double centre_dec_deg;
    double fov_ra_deg;
    double fov_dec_deg;
    double time_start_mjd_utc;
    double time_inc_sec;
    double freq_start_hz;
    double freq_inc_hz;
    double mean;
    double variance;
    double min;
    double max;
    double rms;
};
typedef struct oskar_Image oskar_Image;

/* To maintain binary compatibility, do not change the values
 * in the lists below. */
enum {
    OSKAR_IMAGE_TAG_IMAGE_DATA = 0,
    OSKAR_IMAGE_TAG_DATA_TYPE = 1,
    OSKAR_IMAGE_TAG_DIMENSION_ORDER = 2,
    OSKAR_IMAGE_TAG_NUM_PIXELS_WIDTH = 3,
    OSKAR_IMAGE_TAG_NUM_PIXELS_HEIGHT = 4,
    OSKAR_IMAGE_TAG_NUM_POLS = 5,
    OSKAR_IMAGE_TAG_NUM_TIMES = 6,
    OSKAR_IMAGE_TAG_NUM_CHANNELS = 7,
    OSKAR_IMAGE_TAG_CENTRE_RA = 8,
    OSKAR_IMAGE_TAG_CENTRE_DEC = 9,
    OSKAR_IMAGE_TAG_FOV_RA = 10,
    OSKAR_IMAGE_TAG_FOV_DEC = 11,
    OSKAR_IMAGE_TAG_TIME_START_MJD_UTC = 12,
    OSKAR_IMAGE_TAG_TIME_INC_SEC = 13,
    OSKAR_IMAGE_TAG_FREQ_START_HZ = 14,
    OSKAR_IMAGE_TAG_FREQ_INC_HZ = 15,
    OSKAR_IMAGE_TAG_MEAN = 16,
    OSKAR_IMAGE_TAG_VARIANCE = 17,
    OSKAR_IMAGE_TAG_MIN = 18,
    OSKAR_IMAGE_TAG_MAX = 19,
    OSKAR_IMAGE_TAG_RMS = 20,
    OSKAR_IMAGE_TAG_POL_TYPE = 21
};

/* Do not change the values below - these are merely dimension labels, not the
 * actual dimension order. */
enum {
    OSKAR_IMAGE_DIM_RA = 0,
    OSKAR_IMAGE_DIM_DEC = 1,
    OSKAR_IMAGE_DIM_POL = 2,
    OSKAR_IMAGE_DIM_TIME = 3,
    OSKAR_IMAGE_DIM_CHANNEL = 4
};

enum {
    OSKAR_IMAGE_POL_TYPE_NONE = 0,
    OSKAR_IMAGE_POL_TYPE_STOKES = 1
};

#endif /* OSKAR_IMAGE_H_ */
