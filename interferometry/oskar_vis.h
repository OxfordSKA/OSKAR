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

#ifndef OSKAR_VIS_H_
#define OSKAR_VIS_H_

/**
 * @file oskar_vis.h
 *
 * @deprecated
 * The oskar_Vis structure is deprecated.
 * Do not use these functions or enumerators in new code.
 */

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_Vis;
#ifndef OSKAR_VIS_TYPEDEF_
#define OSKAR_VIS_TYPEDEF_
typedef struct oskar_Vis oskar_Vis;
#endif /* OSKAR_VIS_TYPEDEF_ */

/* To maintain binary compatibility, do not change the values
 * in the lists below. */
enum OSKAR_VIS_TAGS
{
    OSKAR_VIS_TAG_NUM_CHANNELS = 1,
    OSKAR_VIS_TAG_NUM_TIMES = 2,
    OSKAR_VIS_TAG_NUM_BASELINES = 3,
    OSKAR_VIS_TAG_DIMENSION_ORDER = 4,
    OSKAR_VIS_TAG_COORD_TYPE = 5,
    OSKAR_VIS_TAG_AMP_TYPE = 6,
    OSKAR_VIS_TAG_FREQ_START_HZ = 7,
    OSKAR_VIS_TAG_FREQ_INC_HZ = 8,
    OSKAR_VIS_TAG_TIME_START_MJD_UTC = 9,
    OSKAR_VIS_TAG_TIME_INC_SEC = 10,
    OSKAR_VIS_TAG_POL_TYPE = 11,
    OSKAR_VIS_TAG_BASELINE_COORD_UNIT = 12,
    OSKAR_VIS_TAG_BASELINE_UU = 13,
    OSKAR_VIS_TAG_BASELINE_VV = 14,
    OSKAR_VIS_TAG_BASELINE_WW = 15,
    OSKAR_VIS_TAG_AMPLITUDE = 16,
    OSKAR_VIS_TAG_PHASE_CENTRE_RA = 17,
    OSKAR_VIS_TAG_PHASE_CENTRE_DEC = 18,
    OSKAR_VIS_TAG_NUM_STATIONS = 19,
    OSKAR_VIS_TAG_STATION_COORD_UNIT = 20,
    OSKAR_VIS_TAG_STATION_X_OFFSET_ECEF = 21,
    OSKAR_VIS_TAG_STATION_Y_OFFSET_ECEF = 22,
    OSKAR_VIS_TAG_STATION_Z_OFFSET_ECEF = 23,
    OSKAR_VIS_TAG_CHANNEL_BANDWIDTH_HZ = 24,
    OSKAR_VIS_TAG_TIME_AVERAGE_SEC = 25,
    OSKAR_VIS_TAG_TELESCOPE_PATH = 26,
    OSKAR_VIS_TAG_STATION_LON = 27,
    OSKAR_VIS_TAG_STATION_LAT = 28,
    OSKAR_VIS_TAG_STATION_ORIENTATION_X = 29,
    OSKAR_VIS_TAG_STATION_ORIENTATION_Y = 30,
    OSKAR_VIS_TAG_TELESCOPE_LON = 31,
    OSKAR_VIS_TAG_TELESCOPE_LAT = 32,
    OSKAR_VIS_TAG_TELESCOPE_ALT = 33,
    OSKAR_VIS_TAG_STATION_X_ENU = 34,
    OSKAR_VIS_TAG_STATION_Y_ENU = 35,
    OSKAR_VIS_TAG_STATION_Z_ENU = 36
};

/* Do not change the values below - these are merely dimension labels, not the
 * actual dimension order. */
enum OSKAR_VIS_DIM
{
    OSKAR_VIS_DIM_CHANNEL = 0,
    OSKAR_VIS_DIM_TIME = 1,
    OSKAR_VIS_DIM_BASELINE = 2,
    OSKAR_VIS_DIM_POLARISATION = 3
};

enum OSKAR_VIS_POL_TYPE
{
    OSKAR_VIS_POL_TYPE_NONE = 0,
    OSKAR_VIS_POL_TYPE_LINEAR = 1
};

enum OSKAR_VIS_BASELINE_COORD_UNIT
{
    OSKAR_VIS_BASELINE_COORD_UNIT_METRES = 1
};

enum OSKAR_VIS_STATION_COORD_UNIT
{
    OSKAR_VIS_STATION_COORD_UNIT_METRES = 1
};

#ifdef __cplusplus
}
#endif

#include <oskar_vis_accessors.h>
#include <oskar_vis_create.h>
#include <oskar_vis_free.h>
#include <oskar_vis_read.h>
#include <oskar_vis_write.h>

#endif /* OSKAR_VIS_H_ */
