/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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
    OSKAR_VIS_HEADER_TAG_STATION_Z_OFFSET_ECEF    = 34,
    OSKAR_VIS_HEADER_TAG_ELEMENT_X_ENU            = 35,
    OSKAR_VIS_HEADER_TAG_ELEMENT_Y_ENU            = 36,
    OSKAR_VIS_HEADER_TAG_ELEMENT_Z_ENU            = 37
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

#include <vis/oskar_vis_header_accessors.h>
#include <vis/oskar_vis_header_create.h>
#include <vis/oskar_vis_header_free.h>
#include <vis/oskar_vis_header_read.h>
#include <vis/oskar_vis_header_write.h>
#include <vis/oskar_vis_header_write_ms.h>

#endif /* include guard */
