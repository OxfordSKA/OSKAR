/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "binary/oskar_binary.h"
#include "mem/oskar_mem.h"
#include "sky/oskar_sky.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_get_binary_tag_string.h"
#include "vis/oskar_vis_header.h"
#include "vis/oskar_vis_block.h"
#include <cstdlib>


TEST(get_binary_tag_string, metadata)
{
    ASSERT_STREQ("Date",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_METADATA,
                    OSKAR_TAG_METADATA_DATE_TIME_STRING
            )
    );
    ASSERT_STREQ("OSKAR version",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_METADATA,
                    OSKAR_TAG_METADATA_OSKAR_VERSION_STRING
            )
    );
    ASSERT_STREQ("Username",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_METADATA, OSKAR_TAG_METADATA_USERNAME
            )
    );
    ASSERT_STREQ("Working directory",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_METADATA, OSKAR_TAG_METADATA_CWD
            )
    );
}


TEST(get_binary_tag_string, settings)
{
    ASSERT_STREQ("Settings file path",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_SETTINGS, OSKAR_TAG_SETTINGS_PATH
            )
    );
    ASSERT_STREQ("Settings file",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_SETTINGS, OSKAR_TAG_SETTINGS
            )
    );
}


TEST(get_binary_tag_string, run)
{
    ASSERT_STREQ("Run log",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_RUN, OSKAR_TAG_RUN_LOG
            )
    );
}


TEST(get_binary_tag_string, vis_header)
{
    ASSERT_STREQ("Telescope model path",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_TELESCOPE_PATH
            )
    );
    ASSERT_STREQ("Number of tags per block",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_NUM_TAGS_PER_BLOCK
            )
    );
    ASSERT_STREQ("Auto-correlations present",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_WRITE_AUTO_CORRELATIONS
            )
    );
    ASSERT_STREQ("Cross-correlations present",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_WRITE_CROSS_CORRELATIONS
            )
    );
    ASSERT_STREQ("Visibility amplitude type",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_AMP_TYPE
            )
    );
    ASSERT_STREQ("Coordinate precision",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_COORD_PRECISION
            )
    );
    ASSERT_STREQ("Maximum number of times per block",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_MAX_TIMES_PER_BLOCK
            )
    );
    ASSERT_STREQ("Number of times total",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_NUM_TIMES_TOTAL
            )
    );
    ASSERT_STREQ("Maximum number of channels per block",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_MAX_CHANNELS_PER_BLOCK
            )
    );
    ASSERT_STREQ("Number of channels total",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_NUM_CHANNELS_TOTAL
            )
    );
    ASSERT_STREQ("Number of stations",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_NUM_STATIONS
            )
    );
    ASSERT_STREQ("Polarisation type",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_POL_TYPE
            )
    );
    ASSERT_STREQ("Phase centre coordinate type",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_PHASE_CENTRE_COORD_TYPE
            )
    );
    ASSERT_STREQ("Phase centre [deg]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_PHASE_CENTRE_DEG
            )
    );
    ASSERT_STREQ("Frequency start [Hz]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_FREQ_START_HZ
            )
    );
    ASSERT_STREQ("Frequency inc [Hz]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_FREQ_INC_HZ
            )
    );
    ASSERT_STREQ("Channel bandwidth [Hz]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_CHANNEL_BANDWIDTH_HZ
            )
    );
    ASSERT_STREQ("Start time [MJD, UTC]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_TIME_START_MJD_UTC
            )
    );
    ASSERT_STREQ("Time inc [s]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_TIME_INC_SEC
            )
    );
    ASSERT_STREQ("Time average integration [s]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_TIME_AVERAGE_SEC
            )
    );
    ASSERT_STREQ("Telescope longitude [deg]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_TELESCOPE_REF_LON_DEG
            )
    );
    ASSERT_STREQ("Telescope latitude [deg]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_TELESCOPE_REF_LAT_DEG
            )
    );
    ASSERT_STREQ("Telescope altitude [m]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_TELESCOPE_REF_ALT_M
            )
    );
    ASSERT_STREQ("Station X coordinates (offset ECEF) [m]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_STATION_X_OFFSET_ECEF
            )
    );
    ASSERT_STREQ("Station Y coordinates (offset ECEF) [m]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_STATION_Y_OFFSET_ECEF
            )
    );
    ASSERT_STREQ("Station Z coordinates (offset ECEF) [m]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_STATION_Z_OFFSET_ECEF
            )
    );
    ASSERT_STREQ("Element X coordinates (ENU) [m]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_ELEMENT_X_ENU
            )
    );
    ASSERT_STREQ("Element Y coordinates (ENU) [m]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_ELEMENT_Y_ENU
            )
    );
    ASSERT_STREQ("Element Z coordinates (ENU) [m]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_ELEMENT_Z_ENU
            )
    );
    ASSERT_STREQ("Station name",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_STATION_NAME
            )
    );
    ASSERT_STREQ("Station diameter [m]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_STATION_DIAMETER
            )
    );
    ASSERT_STREQ("Element Euler angle (X dipole, alpha) [rad]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_ELEMENT_FEED_ANGLE_X_A
            )
    );
    ASSERT_STREQ("Element Euler angle (Y dipole, alpha) [rad]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_ELEMENT_FEED_ANGLE_Y_A
            )
    );
    ASSERT_STREQ("Element Euler angle (X dipole, beta) [rad]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_ELEMENT_FEED_ANGLE_X_B
            )
    );
    ASSERT_STREQ("Element Euler angle (Y dipole, beta) [rad]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_ELEMENT_FEED_ANGLE_Y_B
            )
    );
    ASSERT_STREQ("Element Euler angle (X dipole, gamma) [rad]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_ELEMENT_FEED_ANGLE_X_C
            )
    );
    ASSERT_STREQ("Element Euler angle (Y dipole, gamma) [rad]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_HEADER,
                    OSKAR_VIS_HEADER_TAG_ELEMENT_FEED_ANGLE_Y_C
            )
    );
}


TEST(get_binary_tag_string, vis_block)
{
    ASSERT_STREQ("Dimension start and size",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_BLOCK,
                    OSKAR_VIS_BLOCK_TAG_DIM_START_AND_SIZE
            )
    );
    ASSERT_STREQ("Auto-correlation data",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_BLOCK,
                    OSKAR_VIS_BLOCK_TAG_AUTO_CORRELATIONS
            )
    );
    ASSERT_STREQ("Cross-correlation data",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_BLOCK,
                    OSKAR_VIS_BLOCK_TAG_CROSS_CORRELATIONS
            )
    );
    ASSERT_STREQ("Baseline UU coordinates [m]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_BLOCK, OSKAR_VIS_BLOCK_TAG_BASELINE_UU
            )
    );
    ASSERT_STREQ("Baseline VV coordinates [m]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_BLOCK, OSKAR_VIS_BLOCK_TAG_BASELINE_VV
            )
    );
    ASSERT_STREQ("Baseline WW coordinates [m]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_BLOCK, OSKAR_VIS_BLOCK_TAG_BASELINE_WW
            )
    );
    ASSERT_STREQ("Station U coordinates [m]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_BLOCK, OSKAR_VIS_BLOCK_TAG_STATION_U
            )
    );
    ASSERT_STREQ("Station V coordinates [m]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_BLOCK, OSKAR_VIS_BLOCK_TAG_STATION_V
            )
    );
    ASSERT_STREQ("Station W coordinates [m]",
            oskar_get_binary_tag_string(
                    OSKAR_TAG_GROUP_VIS_BLOCK, OSKAR_VIS_BLOCK_TAG_STATION_W
            )
    );
}
