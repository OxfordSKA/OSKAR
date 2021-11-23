/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "vis/private_vis_header.h"
#include "vis/oskar_vis_header.h"
#include "mem/oskar_binary_write_mem.h"
#include "utility/oskar_dir.h"
#include "oskar_version.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Binary* oskar_vis_header_write(const oskar_VisHeader* hdr,
        const char* filename, int* status)
{
    int i = 0;
    unsigned char grp = OSKAR_TAG_GROUP_VIS_HEADER;
    oskar_Binary* h = 0;
    char *str = 0, time_str[80];
    struct tm* timeinfo = 0;
    if (*status) return 0;

    /* Create the file handle. */
    h = oskar_binary_create(filename, 'w', status);
    if (*status)
    {
        oskar_binary_free(h);
        return 0;
    }

    /* Write the system date and time. */
    const time_t unix_time = time(0);
    timeinfo = localtime(&unix_time);
    strftime(time_str, sizeof(time_str), "%Y-%m-%d, %H:%M:%S (%Z)", timeinfo);
    oskar_binary_write(h, OSKAR_CHAR,
            OSKAR_TAG_GROUP_METADATA, OSKAR_TAG_METADATA_DATE_TIME_STRING,
            0, 1 + strlen(time_str), time_str, status);

    /* Write the OSKAR version string. */
    oskar_binary_write(h, OSKAR_CHAR,
            OSKAR_TAG_GROUP_METADATA, OSKAR_TAG_METADATA_OSKAR_VERSION_STRING,
            0, 1 + strlen(OSKAR_VERSION_STR), OSKAR_VERSION_STR, status);

    /* Write the current working directory. */
    str = oskar_dir_cwd();
    if (str)
    {
        oskar_binary_write(h, OSKAR_CHAR,
                OSKAR_TAG_GROUP_METADATA, OSKAR_TAG_METADATA_CWD,
                0, 1 + strlen(str), str, status);
    }
    free(str);

    /* Write the username. */
    str = getenv("USERNAME");
    if (!str) str = getenv("USER");
    if (str && strlen(str) > 0)
    {
        oskar_binary_write(h, OSKAR_CHAR,
                OSKAR_TAG_GROUP_METADATA, OSKAR_TAG_METADATA_USERNAME,
                0, 1 + strlen(str), str, status);
    }

    /* If settings exist, write out the data. */
    str = oskar_mem_char(hdr->settings);
    if (str && strlen(str) > 0)
    {
        oskar_binary_write_mem(h, hdr->settings,
                OSKAR_TAG_GROUP_SETTINGS, OSKAR_TAG_SETTINGS, 0, 0, status);
    }

    /* Write the telescope model path. */
    oskar_binary_write_mem(h, hdr->telescope_path, grp,
            OSKAR_VIS_HEADER_TAG_TELESCOPE_PATH, 0, 0, status);

    /* Write the number of binary tags per block. */
    oskar_binary_write_int(h, grp,
            OSKAR_VIS_HEADER_TAG_NUM_TAGS_PER_BLOCK, 0,
            hdr->num_tags_per_block, status);

    /* Write dimensions. */
    oskar_binary_write_int(h, grp,
            OSKAR_VIS_HEADER_TAG_WRITE_AUTO_CORRELATIONS, 0,
            hdr->write_autocorr, status);
    oskar_binary_write_int(h, grp,
            OSKAR_VIS_HEADER_TAG_WRITE_CROSS_CORRELATIONS, 0,
            hdr->write_crosscorr, status);
    oskar_binary_write_int(h, grp,
            OSKAR_VIS_HEADER_TAG_AMP_TYPE, 0,
            hdr->amp_type, status);
    oskar_binary_write_int(h, grp,
            OSKAR_VIS_HEADER_TAG_COORD_PRECISION, 0,
            hdr->coord_precision, status);
    oskar_binary_write_int(h, grp,
            OSKAR_VIS_HEADER_TAG_MAX_TIMES_PER_BLOCK, 0,
            hdr->max_times_per_block, status);
    oskar_binary_write_int(h, grp,
            OSKAR_VIS_HEADER_TAG_NUM_TIMES_TOTAL, 0,
            hdr->num_times_total, status);
    oskar_binary_write_int(h, grp,
            OSKAR_VIS_HEADER_TAG_MAX_CHANNELS_PER_BLOCK, 0,
            hdr->max_channels_per_block, status);
    oskar_binary_write_int(h, grp,
            OSKAR_VIS_HEADER_TAG_NUM_CHANNELS_TOTAL, 0,
            hdr->num_channels_total, status);
    oskar_binary_write_int(h, grp,
            OSKAR_VIS_HEADER_TAG_NUM_STATIONS, 0, hdr->num_stations, status);

    /* Write other visibility metadata. */
    oskar_binary_write_int(h, grp,
            OSKAR_VIS_HEADER_TAG_POL_TYPE, 0, hdr->pol_type, status);
    oskar_binary_write_int(h, grp,
            OSKAR_VIS_HEADER_TAG_PHASE_CENTRE_COORD_TYPE, 0,
            hdr->phase_centre_type, status);
    oskar_binary_write(h, OSKAR_DOUBLE, grp,
            OSKAR_VIS_HEADER_TAG_PHASE_CENTRE_DEG, 0,
            2 * sizeof(double), hdr->phase_centre_deg, status);
    oskar_binary_write_double(h, grp,
            OSKAR_VIS_HEADER_TAG_FREQ_START_HZ, 0, hdr->freq_start_hz, status);
    oskar_binary_write_double(h, grp,
            OSKAR_VIS_HEADER_TAG_FREQ_INC_HZ, 0, hdr->freq_inc_hz, status);
    oskar_binary_write_double(h, grp,
            OSKAR_VIS_HEADER_TAG_CHANNEL_BANDWIDTH_HZ, 0,
            hdr->channel_bandwidth_hz, status);
    oskar_binary_write_double(h, grp,
            OSKAR_VIS_HEADER_TAG_TIME_START_MJD_UTC, 0,
            hdr->time_start_mjd_utc, status);
    oskar_binary_write_double(h, grp,
            OSKAR_VIS_HEADER_TAG_TIME_INC_SEC, 0, hdr->time_inc_sec, status);
    oskar_binary_write_double(h, grp,
            OSKAR_VIS_HEADER_TAG_TIME_AVERAGE_SEC, 0,
            hdr->time_average_sec, status);
    oskar_binary_write_double(h, grp,
            OSKAR_VIS_HEADER_TAG_TELESCOPE_REF_LON_DEG, 0,
            hdr->telescope_centre_lon_deg, status);
    oskar_binary_write_double(h, grp,
            OSKAR_VIS_HEADER_TAG_TELESCOPE_REF_LAT_DEG, 0,
            hdr->telescope_centre_lat_deg, status);
    oskar_binary_write_double(h, grp,
            OSKAR_VIS_HEADER_TAG_TELESCOPE_REF_ALT_M, 0,
            hdr->telescope_centre_alt_m, status);

    /* Write the station coordinates. */
    oskar_binary_write_mem(h, hdr->station_offset_ecef_metres[0], grp,
            OSKAR_VIS_HEADER_TAG_STATION_X_OFFSET_ECEF, 0, 0, status);
    oskar_binary_write_mem(h, hdr->station_offset_ecef_metres[1], grp,
            OSKAR_VIS_HEADER_TAG_STATION_Y_OFFSET_ECEF, 0, 0, status);
    oskar_binary_write_mem(h, hdr->station_offset_ecef_metres[2], grp,
            OSKAR_VIS_HEADER_TAG_STATION_Z_OFFSET_ECEF, 0, 0, status);

    /* Write the station element coordinates. */
    for (i = 0; i < hdr->num_stations; ++i)
    {
        oskar_binary_write_mem(h, hdr->element_enu_metres[0][i], grp,
                OSKAR_VIS_HEADER_TAG_ELEMENT_X_ENU, i, 0, status);
        oskar_binary_write_mem(h, hdr->element_enu_metres[1][i], grp,
                OSKAR_VIS_HEADER_TAG_ELEMENT_Y_ENU, i, 0, status);
        oskar_binary_write_mem(h, hdr->element_enu_metres[2][i], grp,
                OSKAR_VIS_HEADER_TAG_ELEMENT_Z_ENU, i, 0, status);
    }

    return h;
}

#ifdef __cplusplus
}
#endif
