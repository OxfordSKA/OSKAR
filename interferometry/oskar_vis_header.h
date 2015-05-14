/*
 * Copyright (c) 2015, The University of Oxford
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

#ifndef OSKAR_VIS_HEADER_H_
#define OSKAR_VIS_HEADER_H_

/**
 * @file oskar_vis_header.h
 */

/* Public interface. */

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_VisHeader;
#ifndef OSKAR_VIS_HEADER_TYPEDEF_
#define OSKAR_VIS_HEADER_TYPEDEF_
typedef struct oskar_VisHeader oskar_VisHeader;
#endif /* OSKAR_VIS_HEADER_TYPEDEF_ */

/* To maintain binary compatibility, do not change the values
 * in the list below. */
enum OSKAR_VIS_HEADER_TAGS
{
    OSKAR_VIS_HEADER_TAG_TELESCOPE_PATH           = 1,
    OSKAR_VIS_HEADER_TAG_NUM_TAGS_PER_BLOCK       = 2,
    OSKAR_VIS_HEADER_TAG_WRITE_AUTO_CORRELATIONS  = 3,
    OSKAR_VIS_HEADER_TAG_WRITE_CROSS_CORRELATIONS = 4,
    OSKAR_VIS_HEADER_TAG_AMP_TYPE                 = 5,
    OSKAR_VIS_HEADER_TAG_COORD_PRECISION          = 6,
    OSKAR_VIS_HEADER_TAG_MAX_TIMES_PER_BLOCK      = 7,
    OSKAR_VIS_HEADER_TAG_NUM_TIMES_TOTAL          = 8,
    OSKAR_VIS_HEADER_TAG_MAX_CHANNELS_PER_BLOCK   = 9,
    OSKAR_VIS_HEADER_TAG_NUM_CHANNELS_TOTAL       = 10,
    OSKAR_VIS_HEADER_TAG_NUM_STATIONS             = 11,
    OSKAR_VIS_HEADER_TAG_POL_TYPE                 = 12,
    /* Tags 13-20 are reserved for future use. */
    OSKAR_VIS_HEADER_TAG_PHASE_CENTRE_COORD_TYPE  = 21,
    OSKAR_VIS_HEADER_TAG_PHASE_CENTRE_DEG         = 22,
    OSKAR_VIS_HEADER_TAG_FREQ_START_HZ            = 23,
    OSKAR_VIS_HEADER_TAG_FREQ_INC_HZ              = 24,
    OSKAR_VIS_HEADER_TAG_CHANNEL_BANDWIDTH_HZ     = 25,
    OSKAR_VIS_HEADER_TAG_TIME_START_MJD_UTC       = 26,
    OSKAR_VIS_HEADER_TAG_TIME_INC_SEC             = 27,
    OSKAR_VIS_HEADER_TAG_TIME_AVERAGE_SEC         = 28,
    OSKAR_VIS_HEADER_TAG_TELESCOPE_REF_LON_DEG    = 29,
    OSKAR_VIS_HEADER_TAG_TELESCOPE_REF_LAT_DEG    = 30,
    OSKAR_VIS_HEADER_TAG_TELESCOPE_REF_ALT_M      = 31,
    OSKAR_VIS_HEADER_TAG_STATION_X_OFFSET_ECEF    = 32,
    OSKAR_VIS_HEADER_TAG_STATION_Y_OFFSET_ECEF    = 33,
    OSKAR_VIS_HEADER_TAG_STATION_Z_OFFSET_ECEF    = 34
};

enum OSKAR_VIS_HEADER_POL_TYPE
{
    OSKAR_VIS_POL_TYPE_STOKES_I_Q_U_V     =  0,
    OSKAR_VIS_POL_TYPE_STOKES_I           =  1,
    OSKAR_VIS_POL_TYPE_STOKES_Q           =  2,
    OSKAR_VIS_POL_TYPE_STOKES_U           =  3,
    OSKAR_VIS_POL_TYPE_STOKES_V           =  4,
    OSKAR_VIS_POL_TYPE_LINEAR_XX_XY_YX_YY = 10,
    OSKAR_VIS_POL_TYPE_LINEAR_XX          = 11,
    OSKAR_VIS_POL_TYPE_LINEAR_XY          = 12,
    OSKAR_VIS_POL_TYPE_LINEAR_YX          = 13,
    OSKAR_VIS_POL_TYPE_LINEAR_YY          = 14
};

#ifdef __cplusplus
}
#endif

#include <oskar_vis_header_accessors.h>
#include <oskar_vis_header_create.h>
#include <oskar_vis_header_create_copy.h>
#include <oskar_vis_header_free.h>
#include <oskar_vis_header_read.h>
#include <oskar_vis_header_write.h>

#endif /* OSKAR_VIS_HEADER_H_ */
