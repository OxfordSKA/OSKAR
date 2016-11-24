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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "binary/oskar_binary.h"

static void write_test_vis(const char* filename);
static void read_test_vis(const char* filename);

int main(void)
{
    const char* filename = "test.vis";

    /* Write a test visibility data file. */
    write_test_vis(filename);

    /* Read the test visibility data file. */
    read_test_vis(filename);

    return 0;
}


/* To maintain binary compatibility,
 * do not change the numbers in the lists below. */
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

enum OSKAR_VIS_BLOCK_TAGS
{
    OSKAR_VIS_BLOCK_TAG_DIM_START_AND_SIZE    = 1,
    OSKAR_VIS_BLOCK_TAG_AUTO_CORRELATIONS     = 2,
    OSKAR_VIS_BLOCK_TAG_CROSS_CORRELATIONS    = 3,
    OSKAR_VIS_BLOCK_TAG_BASELINE_UU           = 4,
    OSKAR_VIS_BLOCK_TAG_BASELINE_VV           = 5,
    OSKAR_VIS_BLOCK_TAG_BASELINE_WW           = 6
};

#define FLT sizeof(float)
#define DBL sizeof(double)
#define INT sizeof(int)

static void write_test_vis(const char* filename)
{
    int i = 0, status = 0;
    oskar_Binary* h = 0;
    const unsigned char vis_header_group = OSKAR_TAG_GROUP_VIS_HEADER;
    const unsigned char vis_block_group = OSKAR_TAG_GROUP_VIS_BLOCK;

    /* Data to write. */
    const char* telescope_model_path = "my_telescope_model";
    int amp_type, coord_precision, num_tags_per_block = 5, max_times_per_block;
    int num_baselines, num_channels_total, num_stations, num_times_total;
    int num_blocks, num_times_baselines, dim_start_and_size[6];
    int phase_centre_coord_type;
    double phase_centre_deg[2];
    double telescope_ref_lon_deg, telescope_ref_lat_deg, telescope_ref_alt_m;
    double freq_start_hz, freq_inc_hz, channel_bandwidth_hz;
    double time_start_mjd_utc, time_inc_sec, time_average_sec;
    void *station_x, *station_y, *station_z, *vis_block, *uu, *vv, *ww;

    /* Set input metadata. */
    amp_type = OSKAR_DOUBLE_COMPLEX_MATRIX;
    coord_precision = OSKAR_DOUBLE;
    max_times_per_block = 10;
    num_times_total = 33;
    num_channels_total = 2;
    num_stations = 4;
    num_baselines = num_stations * (num_stations - 1) / 2;
    phase_centre_coord_type = 0;
    phase_centre_deg[0] = 10.;
    phase_centre_deg[1] = 20.;
    telescope_ref_lon_deg = -30.;
    telescope_ref_lat_deg =  20.;
    telescope_ref_alt_m =  10.;
    freq_start_hz = 100e6;
    freq_inc_hz = 10e6;
    time_start_mjd_utc = 51544.;
    time_inc_sec = 10.;
    channel_bandwidth_hz = 4e3;
    time_average_sec = 0.08;
    num_times_baselines = max_times_per_block * num_baselines;
    num_blocks = (num_times_total + max_times_per_block - 1) /
            max_times_per_block;
    dim_start_and_size[0] = 0;
    dim_start_and_size[1] = 0;
    dim_start_and_size[2] = max_times_per_block;
    dim_start_and_size[3] = num_channels_total;
    dim_start_and_size[4] = num_baselines;
    dim_start_and_size[5] = num_stations;

    /* Create test visibilities and coordinates. */
    station_x = calloc(num_stations, DBL);
    station_y = calloc(num_stations, DBL);
    station_z = calloc(num_stations, DBL);
    uu        = calloc(num_times_baselines, DBL);
    vv        = calloc(num_times_baselines, DBL);
    ww        = calloc(num_times_baselines, DBL);
    vis_block = calloc(num_times_baselines * num_channels_total, DBL*8);
    for (i = 0; i < num_stations; ++i)
    {
        ((double*)station_x)[i] = i + 0.1;
        ((double*)station_y)[i] = i + 0.2;
        ((double*)station_z)[i] = i + 0.3;
    }
    for (i = 0; i < num_times_baselines; ++i)
    {
        ((double*)uu)[i] = i + 0.11;
        ((double*)vv)[i] = i + 0.22;
        ((double*)ww)[i] = i + 0.33;
    }
    for (i = 0; i < num_times_baselines * num_channels_total; ++i)
    {
        ((double*)vis_block)[8 * i + 0] = i + 0.111; /* XX.re */
        ((double*)vis_block)[8 * i + 1] = i + 0.222; /* XX.im */
        ((double*)vis_block)[8 * i + 2] = i + 0.333; /* XY.re */
        ((double*)vis_block)[8 * i + 3] = i + 0.444; /* XY.im */
        ((double*)vis_block)[8 * i + 4] = i + 0.555; /* YX.re */
        ((double*)vis_block)[8 * i + 5] = i + 0.666; /* YX.im */
        ((double*)vis_block)[8 * i + 6] = i + 0.777; /* YY.re */
        ((double*)vis_block)[8 * i + 7] = i + 0.888; /* YY.im */
    }

    /* Open the test file for writing. */
    printf("Writing test file...\n");
    h = oskar_binary_create(filename, 'w', &status);
    oskar_binary_write(h, OSKAR_CHAR, vis_header_group,
            OSKAR_VIS_HEADER_TAG_TELESCOPE_PATH, 0,
            1 + strlen(telescope_model_path), telescope_model_path, &status);

    /* Write number of tags per block. */
    oskar_binary_write_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_NUM_TAGS_PER_BLOCK, 0, num_tags_per_block,
            &status);

    /* Write dimensions, data types, and options. */
    oskar_binary_write_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_WRITE_AUTO_CORRELATIONS, 0, 0, &status);
    oskar_binary_write_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_WRITE_CROSS_CORRELATIONS, 0, 1, &status);
    oskar_binary_write_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_AMP_TYPE, 0, amp_type, &status);
    oskar_binary_write_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_COORD_PRECISION, 0, coord_precision, &status);
    oskar_binary_write_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_MAX_TIMES_PER_BLOCK, 0,
            max_times_per_block, &status);
    oskar_binary_write_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_NUM_TIMES_TOTAL, 0, num_times_total, &status);
    oskar_binary_write_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_MAX_CHANNELS_PER_BLOCK, 0, num_channels_total,
            &status);
    oskar_binary_write_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_NUM_CHANNELS_TOTAL, 0, num_channels_total,
            &status);
    oskar_binary_write_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_NUM_STATIONS, 0, num_stations, &status);

    /* Write other visibility metadata. */
    oskar_binary_write_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_PHASE_CENTRE_COORD_TYPE, 0,
            phase_centre_coord_type, &status);
    oskar_binary_write(h, OSKAR_DOUBLE, vis_header_group,
            OSKAR_VIS_HEADER_TAG_PHASE_CENTRE_DEG, 0,
            DBL*2, phase_centre_deg, &status);
    oskar_binary_write_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_FREQ_START_HZ, 0, freq_start_hz, &status);
    oskar_binary_write_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_FREQ_INC_HZ, 0, freq_inc_hz, &status);
    oskar_binary_write_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_CHANNEL_BANDWIDTH_HZ, 0,
            channel_bandwidth_hz, &status);
    oskar_binary_write_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_TIME_START_MJD_UTC, 0,
            time_start_mjd_utc, &status);
    oskar_binary_write_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_TIME_INC_SEC, 0, time_inc_sec, &status);
    oskar_binary_write_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_TIME_AVERAGE_SEC, 0,
            time_average_sec, &status);
    oskar_binary_write_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_TELESCOPE_REF_LON_DEG, 0,
            telescope_ref_lon_deg, &status);
    oskar_binary_write_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_TELESCOPE_REF_LAT_DEG, 0,
            telescope_ref_lat_deg, &status);
    oskar_binary_write_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_TELESCOPE_REF_ALT_M, 0,
            telescope_ref_alt_m, &status);

    /* Write station coordinates. */
    oskar_binary_write(h, OSKAR_DOUBLE, vis_header_group,
            OSKAR_VIS_HEADER_TAG_STATION_X_OFFSET_ECEF, 0,
            num_stations * DBL, station_x, &status);
    oskar_binary_write(h, OSKAR_DOUBLE, vis_header_group,
            OSKAR_VIS_HEADER_TAG_STATION_Y_OFFSET_ECEF, 0,
            num_stations * DBL, station_y, &status);
    oskar_binary_write(h, OSKAR_DOUBLE, vis_header_group,
            OSKAR_VIS_HEADER_TAG_STATION_Z_OFFSET_ECEF, 0,
            num_stations * DBL, station_z, &status);

    /* Loop over blocks and write each one. */
    for (i = 0; i < num_blocks; ++i)
    {
        int num_times;

        /* Write block metadata. */
        num_times = max_times_per_block;
        if ((i + 1) * max_times_per_block > num_times_total)
            num_times = num_times_total - i * max_times_per_block;
        dim_start_and_size[0] = i * max_times_per_block;
        dim_start_and_size[1] = 0;
        dim_start_and_size[2] = num_times;
        oskar_binary_write(h, OSKAR_INT, vis_block_group,
                OSKAR_VIS_BLOCK_TAG_DIM_START_AND_SIZE, i,
                INT*6, dim_start_and_size, &status);

        /* Write the visibility data. */
        oskar_binary_write(h, amp_type, vis_block_group,
                OSKAR_VIS_BLOCK_TAG_CROSS_CORRELATIONS, i,
                DBL*8 * num_times_baselines * num_channels_total,
                vis_block, &status);

        /* Write the baseline data. */
        oskar_binary_write(h, OSKAR_DOUBLE, vis_block_group,
                OSKAR_VIS_BLOCK_TAG_BASELINE_UU, i,
                DBL * num_times_baselines, uu, &status);
        oskar_binary_write(h, OSKAR_DOUBLE, vis_block_group,
                OSKAR_VIS_BLOCK_TAG_BASELINE_VV, i,
                DBL * num_times_baselines, vv, &status);
        oskar_binary_write(h, OSKAR_DOUBLE, vis_block_group,
                OSKAR_VIS_BLOCK_TAG_BASELINE_WW, i,
                DBL * num_times_baselines, ww, &status);
    }

    /* Close file. */
    oskar_binary_free(h);

    /* Print status message. */
    if (status != 0)
        printf("Failure writing test file.\n");
    else
        printf("Test file written successfully.\n");

    /* Free local arrays. */
    free(station_x);
    free(station_y);
    free(station_z);
    free(uu);
    free(vv);
    free(ww);
    free(vis_block);
}

/*****************************************************************************/

static void read_test_vis(const char* filename)
{
    int i, j, coord_element_size, vis_element_size, vis_precision, status = 0;
    oskar_Binary* h;
    const unsigned char vis_header_group = OSKAR_TAG_GROUP_VIS_HEADER;
    const unsigned char vis_block_group = OSKAR_TAG_GROUP_VIS_BLOCK;

    /* Data to read. */
    int amp_type, coord_precision, max_times_per_block, num_tags_per_block;
    int num_baselines, num_channels, num_stations, num_times_total;
    int num_blocks, num_times_baselines, dim_start_and_size[6];
    int phase_centre_coord_type, write_autocorr, write_crosscorr;
    double phase_centre_deg[2];
    double telescope_ref_lon_deg, telescope_ref_lat_deg, telescope_ref_alt_m;
    double freq_start_hz, freq_inc_hz, channel_bandwidth_hz;
    double time_start_mjd_utc, time_inc_sec, time_average_sec;
    void *station_x, *station_y, *station_z, *vis_block, *uu, *vv, *ww;

    /* Open the test file for reading. */
    printf("Reading test file...\n");
    h = oskar_binary_create(filename, 'r', &status);

    /* Read number of tags per block. */
    oskar_binary_read_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_NUM_TAGS_PER_BLOCK, 0, &num_tags_per_block,
            &status);

    /* Read dimensions, data types, and options. */
    oskar_binary_read_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_WRITE_AUTO_CORRELATIONS, 0,
            &write_autocorr, &status);
    oskar_binary_read_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_WRITE_CROSS_CORRELATIONS, 0,
            &write_crosscorr, &status);
    oskar_binary_read_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_AMP_TYPE, 0, &amp_type, &status);
    oskar_binary_read_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_COORD_PRECISION, 0, &coord_precision, &status);
    oskar_binary_read_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_MAX_TIMES_PER_BLOCK, 0,
            &max_times_per_block, &status);
    oskar_binary_read_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_NUM_TIMES_TOTAL, 0, &num_times_total, &status);
    oskar_binary_read_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_NUM_CHANNELS_TOTAL, 0, &num_channels, &status);
    oskar_binary_read_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_NUM_STATIONS, 0, &num_stations, &status);

    /* Read visibility metadata. */
    oskar_binary_read_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_PHASE_CENTRE_COORD_TYPE, 0,
            &phase_centre_coord_type, &status);
    oskar_binary_read(h, OSKAR_DOUBLE, vis_header_group,
            OSKAR_VIS_HEADER_TAG_PHASE_CENTRE_DEG, 0,
            DBL*2, phase_centre_deg, &status);
    oskar_binary_read_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_FREQ_START_HZ, 0, &freq_start_hz, &status);
    oskar_binary_read_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_FREQ_INC_HZ, 0, &freq_inc_hz, &status);
    oskar_binary_read_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_CHANNEL_BANDWIDTH_HZ, 0,
            &channel_bandwidth_hz, &status);
    oskar_binary_read_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_TIME_START_MJD_UTC, 0,
            &time_start_mjd_utc, &status);
    oskar_binary_read_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_TIME_INC_SEC, 0,
            &time_inc_sec, &status);
    oskar_binary_read_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_TIME_AVERAGE_SEC, 0,
            &time_average_sec, &status);
    oskar_binary_read_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_TELESCOPE_REF_LON_DEG, 0,
            &telescope_ref_lon_deg, &status);
    oskar_binary_read_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_TELESCOPE_REF_LAT_DEG, 0,
            &telescope_ref_lat_deg, &status);
    oskar_binary_read_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_TELESCOPE_REF_ALT_M, 0,
            &telescope_ref_alt_m, &status);

    /* Get element sizes. */
    vis_precision      = oskar_type_precision(amp_type);
    vis_element_size   = (vis_precision == OSKAR_DOUBLE ? DBL : FLT);
    coord_element_size = (coord_precision == OSKAR_DOUBLE ? DBL : FLT);

    /* Read the station coordinates. */
    station_x = calloc(num_stations, coord_element_size);
    station_y = calloc(num_stations, coord_element_size);
    station_z = calloc(num_stations, coord_element_size);
    oskar_binary_read(h, coord_precision, vis_header_group,
            OSKAR_VIS_HEADER_TAG_STATION_X_OFFSET_ECEF, 0,
            num_stations * coord_element_size, station_x, &status);
    oskar_binary_read(h, coord_precision, vis_header_group,
            OSKAR_VIS_HEADER_TAG_STATION_Y_OFFSET_ECEF, 0,
            num_stations * coord_element_size, station_y, &status);
    oskar_binary_read(h, coord_precision, vis_header_group,
            OSKAR_VIS_HEADER_TAG_STATION_Z_OFFSET_ECEF, 0,
            num_stations * coord_element_size, station_z, &status);

    /* Print header data. */
    printf("Max. number of times per block: %d\n", max_times_per_block);
    printf("Total number of times: %d\n", num_times_total);
    printf("Number of stations: %d\n", num_stations);
    if (coord_precision == OSKAR_DOUBLE)
    {
        for (i = 0; i < num_stations; ++i)
        {
            printf("Station[%d] at (%.3f, %.3f, %.3f)\n", i,
                    ((double*)station_x)[i],
                    ((double*)station_y)[i],
                    ((double*)station_z)[i]);
        }
    }
    else if (coord_precision == OSKAR_SINGLE)
    {
        for (i = 0; i < num_stations; ++i)
        {
            printf("Station[%d] at (%.3f, %.3f, %.3f)\n", i,
                    ((float*)station_x)[i],
                    ((float*)station_y)[i],
                    ((float*)station_z)[i]);
        }
    }

    /* Calculate expected number of blocks. */
    num_blocks = (num_times_total + max_times_per_block - 1) /
            max_times_per_block;

    /* Allocate visibility data and baseline coordinate arrays. */
    /* This can be done here, as we already know enough to work out
     * the maximum size of the block. */
    num_baselines = num_stations * (num_stations - 1) / 2;
    num_times_baselines = max_times_per_block * num_baselines;
    uu        = calloc(num_times_baselines, coord_element_size);
    vv        = calloc(num_times_baselines, coord_element_size);
    ww        = calloc(num_times_baselines, coord_element_size);
    vis_block = calloc(num_times_baselines * num_channels, 8*vis_element_size);

    /* Loop over blocks and read each one. */
    for (i = 0; i < num_blocks; ++i)
    {
        int b, c, t, num_times, start_time_idx, start_channel_idx;

        /* Set search start index. */
        oskar_binary_set_query_search_start(h, i * num_tags_per_block, &status);

        /* Read block metadata. */
        oskar_binary_read(h, OSKAR_INT, vis_block_group,
                OSKAR_VIS_BLOCK_TAG_DIM_START_AND_SIZE, i,
                INT*6, dim_start_and_size, &status);

        /* Get the number of times actually in the block. */
        start_time_idx      = dim_start_and_size[0];
        start_channel_idx   = dim_start_and_size[1];
        num_times           = dim_start_and_size[2];

        /* Read the visibility data. */
        oskar_binary_read(h, amp_type, vis_block_group,
                OSKAR_VIS_BLOCK_TAG_CROSS_CORRELATIONS, i,
                coord_element_size*8 * num_times_baselines * num_channels,
                vis_block, &status);

        /* Read the baseline data. */
        oskar_binary_read(h, OSKAR_DOUBLE, vis_block_group,
                OSKAR_VIS_BLOCK_TAG_BASELINE_UU, i,
                coord_element_size * num_times_baselines, uu, &status);
        oskar_binary_read(h, OSKAR_DOUBLE, vis_block_group,
                OSKAR_VIS_BLOCK_TAG_BASELINE_VV, i,
                coord_element_size * num_times_baselines, vv, &status);
        oskar_binary_read(h, OSKAR_DOUBLE, vis_block_group,
                OSKAR_VIS_BLOCK_TAG_BASELINE_WW, i,
                coord_element_size * num_times_baselines, ww, &status);

        /* Check for errors. */
        if (status) break;

        /* Print contents of the block. */
        for (t = 0; t < num_times; ++t)
        {
            double mjd_utc, freq_hz;

            /* Get the actual time of the sample. */
            mjd_utc = time_start_mjd_utc +
                    time_inc_sec * (start_time_idx + t + 0.5) / 86400.0;
            printf("-------- Time %d (%.5f)\n", t, mjd_utc);

            if (coord_precision == OSKAR_DOUBLE)
            {
                const double *u, *v, *w;
                u = (const double*) uu;
                v = (const double*) vv;
                w = (const double*) ww;
                for (b = 0; b < num_baselines; ++b)
                {
                    j = b + num_baselines * t;
                    printf("Baseline %d coordinates: (U, V, W) = "
                            "(%.3f, %.3f, %.3f)\n", b, u[j], v[j], w[j]);
                }
            }
            else if (coord_precision == OSKAR_SINGLE)
            {
                const float *u, *v, *w;
                u = (const float*) uu;
                v = (const float*) vv;
                w = (const float*) ww;
                for (b = 0; b < num_baselines; ++b)
                {
                    j = b + num_baselines * t;
                    printf("Baseline %d coordinates: (U, V, W) = "
                            "(%.3f, %.3f, %.3f)\n", b, u[j], v[j], w[j]);
                }
            }

            for (c = 0; c < num_channels; ++c)
            {
                /* Get the actual frequency of the sample. */
                freq_hz = freq_start_hz + freq_inc_hz * (start_channel_idx + c);
                printf("---------------- Channel %d (%.3f MHz)\n",
                        c, freq_hz / 1e6);

                if (vis_precision == OSKAR_DOUBLE)
                {
                    const double *d;
                    d = (const double*) vis_block;
                    for (b = 0; b < num_baselines; ++b)
                    {
                        j = 8 * (b + num_baselines * (c + num_channels * t));
                        printf("Baseline %d visibility:\n"
                                "            [%.3f, %.3f] [%.3f, %.3f]\n"
                                "            [%.3f, %.3f] [%.3f, %.3f]\n", b,
                                d[j + 0], d[j + 1], d[j + 2], d[j + 3],
                                d[j + 4], d[j + 5], d[j + 6], d[j + 7]);
                    }
                }
                else if (vis_precision == OSKAR_SINGLE)
                {
                    const float *d;
                    d = (const float*) vis_block;
                    for (b = 0; b < num_baselines; ++b)
                    {
                        j = 8 * (b + num_baselines * (c + num_channels * t));
                        printf("Baseline %d visibility:\n"
                                "            [%.3f, %.3f] [%.3f, %.3f]\n"
                                "            [%.3f, %.3f] [%.3f, %.3f]\n", b,
                                d[j + 0], d[j + 1], d[j + 2], d[j + 3],
                                d[j + 4], d[j + 5], d[j + 6], d[j + 7]);
                    }
                }
            }
        }
        /* End printing of visibility block. */
    }

    /* Close the file. */
    oskar_binary_free(h);

    /* Print status message. */
    if (status != 0)
        printf("Failure reading test file.\n");
    else
        printf("Test file read successfully.\n");

    /* Free local arrays. */
    free(station_x);
    free(station_y);
    free(station_z);
    free(uu);
    free(vv);
    free(ww);
    free(vis_block);
}

