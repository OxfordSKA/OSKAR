/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "vis/private_vis_header.h"
#include "vis/oskar_vis_header.h"
#include "mem/oskar_binary_read_mem.h"

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_VisHeader* oskar_vis_header_read(oskar_Binary* h, int* status)
{
    int i = 0;
    int num_channels_total = 0, num_times_total = 0, num_stations = 0;
    int max_channels_per_block = 0, max_times_per_block = 0, tag_error = 0;
    int amp_type = 0, coord_precision = 0;
    int write_crosscorr = 0, write_autocorr = 0;
    int num_tags_header = 0;
    unsigned char grp = OSKAR_TAG_GROUP_VIS_HEADER;
    oskar_VisHeader* vis = 0;
    if (*status) return 0;

    /* Read essential metadata. */
    oskar_binary_read_int(h, grp,
            OSKAR_VIS_HEADER_TAG_WRITE_AUTO_CORRELATIONS,
            0, &write_autocorr, status);
    oskar_binary_read_int(h, grp,
            OSKAR_VIS_HEADER_TAG_WRITE_CROSS_CORRELATIONS,
            0, &write_crosscorr, status);
    oskar_binary_read_int(h, grp, OSKAR_VIS_HEADER_TAG_AMP_TYPE,
            0, &amp_type, status);
    oskar_binary_read_int(h, grp, OSKAR_VIS_HEADER_TAG_COORD_PRECISION,
            0, &coord_precision, status);
    oskar_binary_read_int(h, grp, OSKAR_VIS_HEADER_TAG_MAX_TIMES_PER_BLOCK,
            0, &max_times_per_block, status);
    oskar_binary_read_int(h, grp, OSKAR_VIS_HEADER_TAG_NUM_TIMES_TOTAL,
            0, &num_times_total, status);
    oskar_binary_read_int(h, grp, OSKAR_VIS_HEADER_TAG_MAX_CHANNELS_PER_BLOCK,
            0, &max_channels_per_block, status);
    oskar_binary_read_int(h, grp, OSKAR_VIS_HEADER_TAG_NUM_CHANNELS_TOTAL,
            0, &num_channels_total, status);
    oskar_binary_read_int(h, grp, OSKAR_VIS_HEADER_TAG_NUM_STATIONS,
            0, &num_stations, status);
    num_tags_header += 9;
    if (*status) return 0;

    /* Create the visibility header. */
    vis = oskar_vis_header_create(amp_type, coord_precision,
            max_times_per_block, num_times_total,
            max_channels_per_block, num_channels_total, num_stations,
            write_autocorr, write_crosscorr, status);
    if (*status) return vis;

    /* Read the number of tags per block. */
    oskar_binary_read_int(h, grp, OSKAR_VIS_HEADER_TAG_NUM_TAGS_PER_BLOCK, 0,
            &vis->num_tags_per_block, status);
    num_tags_header += 1;

    /* Optionally read the settings data (ignore the error code). */
    tag_error = 0;
    oskar_binary_read_mem(h, vis->settings,
            OSKAR_TAG_GROUP_SETTINGS, OSKAR_TAG_SETTINGS, 0, &tag_error);
    if (!tag_error) num_tags_header += 1;

    /* Read the telescope model path. */
    oskar_binary_read_mem(h, vis->telescope_path,
            grp, OSKAR_VIS_HEADER_TAG_TELESCOPE_PATH, 0, status);
    num_tags_header += 1;

    /* Read other visibility metadata. */
    oskar_binary_read_int(h, grp, OSKAR_VIS_HEADER_TAG_POL_TYPE, 0,
            &vis->pol_type, status);
    oskar_binary_read_int(h, grp,
            OSKAR_VIS_HEADER_TAG_PHASE_CENTRE_COORD_TYPE, 0,
            &vis->phase_centre_type, status);
    oskar_binary_read(h, OSKAR_DOUBLE, grp,
            OSKAR_VIS_HEADER_TAG_PHASE_CENTRE_DEG, 0,
            2 * sizeof(double), &vis->phase_centre_deg, status);
    oskar_binary_read_double(h, grp, OSKAR_VIS_HEADER_TAG_FREQ_START_HZ, 0,
            &vis->freq_start_hz, status);
    oskar_binary_read_double(h, grp, OSKAR_VIS_HEADER_TAG_FREQ_INC_HZ, 0,
            &vis->freq_inc_hz, status);
    oskar_binary_read_double(h, grp,
            OSKAR_VIS_HEADER_TAG_CHANNEL_BANDWIDTH_HZ, 0,
            &vis->channel_bandwidth_hz, status);
    oskar_binary_read_double(h, grp, OSKAR_VIS_HEADER_TAG_TIME_START_MJD_UTC, 0,
            &vis->time_start_mjd_utc, status);
    oskar_binary_read_double(h, grp, OSKAR_VIS_HEADER_TAG_TIME_INC_SEC, 0,
            &vis->time_inc_sec, status);
    oskar_binary_read_double(h, grp,
            OSKAR_VIS_HEADER_TAG_TIME_AVERAGE_SEC, 0,
            &vis->time_average_sec, status);
    oskar_binary_read_double(h, grp,
            OSKAR_VIS_HEADER_TAG_TELESCOPE_REF_LON_DEG, 0,
            &vis->telescope_centre_lon_deg, status);
    oskar_binary_read_double(h, grp,
            OSKAR_VIS_HEADER_TAG_TELESCOPE_REF_LAT_DEG, 0,
            &vis->telescope_centre_lat_deg, status);
    oskar_binary_read_double(h, grp,
            OSKAR_VIS_HEADER_TAG_TELESCOPE_REF_ALT_M, 0,
            &vis->telescope_centre_alt_m, status);
    num_tags_header += 12;

    /* Read the station coordinates. */
    oskar_binary_read_mem(h, vis->station_offset_ecef_metres[0],
            grp, OSKAR_VIS_HEADER_TAG_STATION_X_OFFSET_ECEF, 0, status);
    oskar_binary_read_mem(h, vis->station_offset_ecef_metres[1],
            grp, OSKAR_VIS_HEADER_TAG_STATION_Y_OFFSET_ECEF, 0, status);
    oskar_binary_read_mem(h, vis->station_offset_ecef_metres[2],
            grp, OSKAR_VIS_HEADER_TAG_STATION_Z_OFFSET_ECEF, 0, status);
    num_tags_header += 3;

    /* Optionally read station element coordinates (ignoring error codes). */
    tag_error = 0;
    for (i = 0; i < vis->num_stations; ++i)
    {
        oskar_binary_read_mem(h, vis->element_enu_metres[0][i],
                grp, OSKAR_VIS_HEADER_TAG_ELEMENT_X_ENU, i, &tag_error);
        oskar_binary_read_mem(h, vis->element_enu_metres[1][i],
                grp, OSKAR_VIS_HEADER_TAG_ELEMENT_Y_ENU, i, &tag_error);
        oskar_binary_read_mem(h, vis->element_enu_metres[2][i],
                grp, OSKAR_VIS_HEADER_TAG_ELEMENT_Z_ENU, i, &tag_error);
    }
    if (!tag_error) num_tags_header += (3 * vis->num_stations);

    /* Keep a record of the number of tags in the header. */
    vis->num_tags_header = num_tags_header;

    return vis;
}

#ifdef __cplusplus
}
#endif
