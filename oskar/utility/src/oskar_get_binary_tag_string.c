/*
 * Copyright (c) 2019-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "utility/oskar_get_binary_tag_string.h"

#include "binary/oskar_binary.h"
#include "sky/oskar_sky.h"
#include "vis/oskar_vis_header.h"
#include "vis/oskar_vis_block.h"

#ifdef __cplusplus
extern "C" {
#endif

const char* oskar_get_binary_tag_string(char group, char tag)
{
    switch (group)
    {
    case OSKAR_TAG_GROUP_METADATA:
    {
        switch (tag)
        {
        case OSKAR_TAG_METADATA_DATE_TIME_STRING:
            return "Date";
        case OSKAR_TAG_METADATA_OSKAR_VERSION_STRING:
            return "OSKAR version";
        case OSKAR_TAG_METADATA_USERNAME:
            return "Username";
        case OSKAR_TAG_METADATA_CWD:
            return "Working directory";
        default:                                          /* LCOV_EXCL_LINE */
            return "Unknown metadata group tag";          /* LCOV_EXCL_LINE */
        }
        break;
    }
    case OSKAR_TAG_GROUP_SETTINGS:
    {
        switch (tag)
        {
        case OSKAR_TAG_SETTINGS_PATH:
            return "Settings file path";
        case OSKAR_TAG_SETTINGS:
            return "Settings file";
        default:                                          /* LCOV_EXCL_LINE */
            return "Unknown settings group tag";          /* LCOV_EXCL_LINE */
        }
        break;
    }
    case OSKAR_TAG_GROUP_RUN:
    {
        switch (tag)
        {
        case OSKAR_TAG_RUN_LOG:
            return "Run log";
        default:                                          /* LCOV_EXCL_LINE */
            return "Unknown run group tag";               /* LCOV_EXCL_LINE */
        }
        break;
    }
    case OSKAR_TAG_GROUP_VIS_HEADER:
    {
        switch (tag)
        {
        case OSKAR_VIS_HEADER_TAG_TELESCOPE_PATH:
            return "Telescope model path";
        case OSKAR_VIS_HEADER_TAG_NUM_TAGS_PER_BLOCK:
            return "Number of tags per block";
        case OSKAR_VIS_HEADER_TAG_WRITE_AUTO_CORRELATIONS:
            return "Auto-correlations present";
        case OSKAR_VIS_HEADER_TAG_WRITE_CROSS_CORRELATIONS:
            return "Cross-correlations present";
        case OSKAR_VIS_HEADER_TAG_AMP_TYPE:
            return "Visibility amplitude type";
        case OSKAR_VIS_HEADER_TAG_COORD_PRECISION:
            return "Coordinate precision";
        case OSKAR_VIS_HEADER_TAG_MAX_TIMES_PER_BLOCK:
            return "Maximum number of times per block";
        case OSKAR_VIS_HEADER_TAG_NUM_TIMES_TOTAL:
            return "Number of times total";
        case OSKAR_VIS_HEADER_TAG_MAX_CHANNELS_PER_BLOCK:
            return "Maximum number of channels per block";
        case OSKAR_VIS_HEADER_TAG_NUM_CHANNELS_TOTAL:
            return "Number of channels total";
        case OSKAR_VIS_HEADER_TAG_NUM_STATIONS:
            return "Number of stations";
        case OSKAR_VIS_HEADER_TAG_POL_TYPE:
            return "Polarisation type";
        case OSKAR_VIS_HEADER_TAG_PHASE_CENTRE_COORD_TYPE:
            return "Phase centre coordinate type";
        case OSKAR_VIS_HEADER_TAG_PHASE_CENTRE_DEG:
            return "Phase centre [deg]";
        case OSKAR_VIS_HEADER_TAG_FREQ_START_HZ:
            return "Frequency start [Hz]";
        case OSKAR_VIS_HEADER_TAG_FREQ_INC_HZ:
            return "Frequency inc [Hz]";
        case OSKAR_VIS_HEADER_TAG_CHANNEL_BANDWIDTH_HZ:
            return "Channel bandwidth [Hz]";
        case OSKAR_VIS_HEADER_TAG_TIME_START_MJD_UTC:
            return "Start time [MJD, UTC]";
        case OSKAR_VIS_HEADER_TAG_TIME_INC_SEC:
            return "Time inc [s]";
        case OSKAR_VIS_HEADER_TAG_TIME_AVERAGE_SEC:
            return "Time average integration [s]";
        case OSKAR_VIS_HEADER_TAG_TELESCOPE_REF_LON_DEG:
            return "Telescope longitude [deg]";
        case OSKAR_VIS_HEADER_TAG_TELESCOPE_REF_LAT_DEG:
            return "Telescope latitude [deg]";
        case OSKAR_VIS_HEADER_TAG_TELESCOPE_REF_ALT_M:
            return "Telescope altitude [m]";
        case OSKAR_VIS_HEADER_TAG_STATION_X_OFFSET_ECEF:
            return "Station X coordinates (offset ECEF) [m]";
        case OSKAR_VIS_HEADER_TAG_STATION_Y_OFFSET_ECEF:
            return "Station Y coordinates (offset ECEF) [m]";
        case OSKAR_VIS_HEADER_TAG_STATION_Z_OFFSET_ECEF:
            return "Station Z coordinates (offset ECEF) [m]";
        case OSKAR_VIS_HEADER_TAG_ELEMENT_X_ENU:
            return "Element X coordinates (ENU) [m]";
        case OSKAR_VIS_HEADER_TAG_ELEMENT_Y_ENU:
            return "Element Y coordinates (ENU) [m]";
        case OSKAR_VIS_HEADER_TAG_ELEMENT_Z_ENU:
            return "Element Z coordinates (ENU) [m]";
        case OSKAR_VIS_HEADER_TAG_STATION_NAME:
            return "Station name";
        case OSKAR_VIS_HEADER_TAG_STATION_DIAMETER:
            return "Station diameter [m]";
        case OSKAR_VIS_HEADER_TAG_ELEMENT_FEED_ANGLE_X_A:
            return "Element Euler angle (X dipole, alpha) [rad]";
        case OSKAR_VIS_HEADER_TAG_ELEMENT_FEED_ANGLE_Y_A:
            return "Element Euler angle (Y dipole, alpha) [rad]";
        case OSKAR_VIS_HEADER_TAG_ELEMENT_FEED_ANGLE_X_B:
            return "Element Euler angle (X dipole, beta) [rad]";
        case OSKAR_VIS_HEADER_TAG_ELEMENT_FEED_ANGLE_Y_B:
            return "Element Euler angle (Y dipole, beta) [rad]";
        case OSKAR_VIS_HEADER_TAG_ELEMENT_FEED_ANGLE_X_C:
            return "Element Euler angle (X dipole, gamma) [rad]";
        case OSKAR_VIS_HEADER_TAG_ELEMENT_FEED_ANGLE_Y_C:
            return "Element Euler angle (Y dipole, gamma) [rad]";
        default:                                          /* LCOV_EXCL_LINE */
            return "Unknown visibility header group tag"; /* LCOV_EXCL_LINE */
        }
        break;
    }
    case OSKAR_TAG_GROUP_VIS_BLOCK:
    {
        switch (tag)
        {
        case OSKAR_VIS_BLOCK_TAG_DIM_START_AND_SIZE:
            return "Dimension start and size";
        case OSKAR_VIS_BLOCK_TAG_AUTO_CORRELATIONS:
            return "Auto-correlation data";
        case OSKAR_VIS_BLOCK_TAG_CROSS_CORRELATIONS:
            return "Cross-correlation data";
        case OSKAR_VIS_BLOCK_TAG_BASELINE_UU:
            return "Baseline UU coordinates [m]";
        case OSKAR_VIS_BLOCK_TAG_BASELINE_VV:
            return "Baseline VV coordinates [m]";
        case OSKAR_VIS_BLOCK_TAG_BASELINE_WW:
            return "Baseline WW coordinates [m]";
        case OSKAR_VIS_BLOCK_TAG_STATION_U:
            return "Station U coordinates [m]";
        case OSKAR_VIS_BLOCK_TAG_STATION_V:
            return "Station V coordinates [m]";
        case OSKAR_VIS_BLOCK_TAG_STATION_W:
            return "Station W coordinates [m]";
        default:                                          /* LCOV_EXCL_LINE */
            return "Unknown visibility block group tag";  /* LCOV_EXCL_LINE */
        }
        break;
    }
    default:                                              /* LCOV_EXCL_LINE */
        break;                                            /* LCOV_EXCL_LINE */
    }
    return "Unknown tag group";                           /* LCOV_EXCL_LINE */
}

#ifdef __cplusplus
}
#endif
