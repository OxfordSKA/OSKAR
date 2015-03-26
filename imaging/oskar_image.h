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

#ifndef OSKAR_IMAGE_H_
#define OSKAR_IMAGE_H_

/**
 * @file oskar_image.h
 *
 * @deprecated
 * The oskar_Image structure is deprecated.
 * Do not use these functions or enumerators in new code.
 */

/* Public interface. */

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_Image;
#ifndef OSKAR_IMAGE_TYPEDEF_
#define OSKAR_IMAGE_TYPEDEF_
typedef struct oskar_Image oskar_Image;
#endif /* OSKAR_IMAGE_TYPEDEF_ */

/* To maintain binary compatibility, do not change the values
 * in the lists below. */
enum OSKAR_IMAGE_TAGS
{
    OSKAR_IMAGE_TAG_IMAGE_DATA = 0,
    OSKAR_IMAGE_TAG_IMAGE_TYPE = 1,
    OSKAR_IMAGE_TAG_DATA_TYPE = 2,
    OSKAR_IMAGE_TAG_DIMENSION_ORDER = 3,
    OSKAR_IMAGE_TAG_NUM_PIXELS_WIDTH = 4,
    OSKAR_IMAGE_TAG_NUM_PIXELS_HEIGHT = 5,
    OSKAR_IMAGE_TAG_NUM_POLS = 6,
    OSKAR_IMAGE_TAG_NUM_TIMES = 7,
    OSKAR_IMAGE_TAG_NUM_CHANNELS = 8,
    OSKAR_IMAGE_TAG_CENTRE_LONGITUDE = 9,
    OSKAR_IMAGE_TAG_CENTRE_LATITUDE = 10,
    OSKAR_IMAGE_TAG_FOV_LONGITUDE = 11,
    OSKAR_IMAGE_TAG_FOV_LATITUDE = 12,
    OSKAR_IMAGE_TAG_TIME_START_MJD_UTC = 13,
    OSKAR_IMAGE_TAG_TIME_INC_SEC = 14,
    OSKAR_IMAGE_TAG_FREQ_START_HZ = 15,
    OSKAR_IMAGE_TAG_FREQ_INC_HZ = 16,
    OSKAR_IMAGE_TAG_MEAN = 17,
    OSKAR_IMAGE_TAG_VARIANCE = 18,
    OSKAR_IMAGE_TAG_MIN = 19,
    OSKAR_IMAGE_TAG_MAX = 20,
    OSKAR_IMAGE_TAG_RMS = 21,
    OSKAR_IMAGE_TAG_GRID_TYPE = 22,    /* NEW v2.4 */
    OSKAR_IMAGE_TAG_COORD_FRAME = 23,  /* NEW v2.4 */
    OSKAR_IMAGE_TAG_HEALPIX_NSIDE = 24 /* NEW v2.4 */
};

/* Do not change the values below - these are merely dimension labels, not the
 * actual dimension order. */
enum OSKAR_IMAGE_DIM
{
    OSKAR_IMAGE_DIM_LONGITUDE = 0,
    OSKAR_IMAGE_DIM_LATITUDE = 1,
    OSKAR_IMAGE_DIM_POL = 2,
    OSKAR_IMAGE_DIM_TIME = 3,
    OSKAR_IMAGE_DIM_CHANNEL = 4
};

enum OSKAR_IMAGE_TYPE
{
    OSKAR_IMAGE_TYPE_UNDEF = 0,

    OSKAR_IMAGE_TYPE_STOKES   = 1, /* IQUV */
    OSKAR_IMAGE_TYPE_STOKES_I = 2,
    OSKAR_IMAGE_TYPE_STOKES_Q = 3,
    OSKAR_IMAGE_TYPE_STOKES_U = 4,
    OSKAR_IMAGE_TYPE_STOKES_V = 5,

    OSKAR_IMAGE_TYPE_POL_LINEAR = 6, /* all linear polarisations XX,XY,YX,YY */
    OSKAR_IMAGE_TYPE_POL_XX     = 7,
    OSKAR_IMAGE_TYPE_POL_YY     = 8,
    OSKAR_IMAGE_TYPE_POL_XY     = 9,
    OSKAR_IMAGE_TYPE_POL_YX     = 10,

    OSKAR_IMAGE_TYPE_PSF        = 50,

    OSKAR_IMAGE_TYPE_BEAM_SCALAR = 100,
    OSKAR_IMAGE_TYPE_BEAM_POLARISED = 101
};

/**
 * @brief Enumerator describing the image pixel grid distribution.
 */
enum OSKAR_IMAGE_GRID_TYPE
{
    OSKAR_IMAGE_GRID_TYPE_UNDEF = 0,
    OSKAR_IMAGE_GRID_TYPE_RECTILINEAR = 1,
    OSKAR_IMAGE_GRID_TYPE_HEALPIX = 2
};

/**
 * @brief Enumerator describing the image coordinate frame.
 */
enum OSKAR_IMAGE_COORD_FRAME
{
    OSKAR_IMAGE_COORD_FRAME_UNDEF = 0,
    OSKAR_IMAGE_COORD_FRAME_EQUATORIAL = 1,
    OSKAR_IMAGE_COORD_FRAME_HORIZON = 2
};

#ifdef __cplusplus
}
#endif

#include <oskar_image_accessors.h>
#include <oskar_image_create.h>
#include <oskar_image_free.h>
#include <oskar_image_resize.h>

#endif /* OSKAR_IMAGE_H_ */
