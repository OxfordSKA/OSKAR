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

#include <oskar_binary.h>

/* To maintain binary compatibility,
 * do not change the numbers in the lists below. */
enum OSKAR_VIS_HEADER_TAGS
{
    OSKAR_VIS_HEADER_TAG_TELESCOPE_PATH          = 1,
    OSKAR_VIS_HEADER_TAG_NUM_TAGS_PER_BLOCK      = 2,
    OSKAR_VIS_HEADER_TAG_WRITE_AUTOCORRELATIONS  = 3,
    OSKAR_VIS_HEADER_TAG_AMP_TYPE                = 4,
    OSKAR_VIS_HEADER_TAG_MAX_TIMES_PER_BLOCK     = 5,
    OSKAR_VIS_HEADER_TAG_NUM_TIMES_TOTAL         = 6,
    OSKAR_VIS_HEADER_TAG_NUM_CHANNELS            = 7,
    OSKAR_VIS_HEADER_TAG_NUM_STATIONS            = 8,
    OSKAR_VIS_HEADER_TAG_FREQ_START_HZ           = 9,
    OSKAR_VIS_HEADER_TAG_FREQ_INC_HZ             = 10,
    OSKAR_VIS_HEADER_TAG_CHANNEL_BANDWIDTH_HZ    = 11,
    OSKAR_VIS_HEADER_TAG_TIME_START_MJD_UTC      = 12,
    OSKAR_VIS_HEADER_TAG_TIME_INC_SEC            = 13,
    OSKAR_VIS_HEADER_TAG_TIME_AVERAGE_SEC        = 14,
    OSKAR_VIS_HEADER_TAG_PHASE_CENTRE            = 15,
    OSKAR_VIS_HEADER_TAG_TELESCOPE_CENTRE        = 16,
    OSKAR_VIS_HEADER_TAG_STATION_X_OFFSET_ECEF   = 17,
    OSKAR_VIS_HEADER_TAG_STATION_Y_OFFSET_ECEF   = 18,
    OSKAR_VIS_HEADER_TAG_STATION_Z_OFFSET_ECEF   = 19
};

enum OSKAR_VIS_BLOCK_TAGS
{
    OSKAR_VIS_BLOCK_TAG_DIM_START_AND_SIZE    = 1,
    OSKAR_VIS_BLOCK_TAG_FREQ_REF_INC_HZ       = 2,
    OSKAR_VIS_BLOCK_TAG_TIME_REF_INC_MJD_UTC  = 3,
    OSKAR_VIS_BLOCK_TAG_AUTO_CORRELATIONS     = 4,
    OSKAR_VIS_BLOCK_TAG_CROSS_CORRELATIONS    = 5,
    OSKAR_VIS_BLOCK_TAG_BASELINE_UU           = 6,
    OSKAR_VIS_BLOCK_TAG_BASELINE_VV           = 7,
    OSKAR_VIS_BLOCK_TAG_BASELINE_WW           = 8
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
    int amp_type, max_times_per_block, num_tags_per_block = 7;
    int num_baselines, num_channels, num_stations, num_times_total;
    int num_blocks, num_times_baselines, dim_start_and_size[6];
    double phase_centre[2], telescope_centre[3];
    double freq_start_inc[2], time_start_inc[2];
    double channel_bandwidth_hz, time_average_sec;
    void *station_x, *station_y, *station_z, *vis_block, *uu, *vv, *ww;

    /* Set input metadata. */
    amp_type = OSKAR_DOUBLE_COMPLEX_MATRIX;
    max_times_per_block = 10;
    num_times_total = 33;
    num_channels = 2;
    num_stations = 4;
    num_baselines = num_stations * (num_stations - 1) / 2;
    phase_centre[0] = 10.;
    phase_centre[1] = 20.;
    telescope_centre[0] = -30.;
    telescope_centre[1] =  20.;
    telescope_centre[2] =  10.;
    freq_start_inc[0] = 100e6;
    freq_start_inc[1] = 10e6;
    time_start_inc[0] = 51544.;
    time_start_inc[1] = 0.01;
    channel_bandwidth_hz = 4e3;
    time_average_sec = 0.08;
    num_times_baselines = max_times_per_block * num_baselines;
    num_blocks = (num_times_total + max_times_per_block - 1) /
            max_times_per_block;
    dim_start_and_size[0] = 0;
    dim_start_and_size[1] = 0;
    dim_start_and_size[2] = max_times_per_block;
    dim_start_and_size[3] = num_channels;
    dim_start_and_size[4] = num_baselines;
    dim_start_and_size[5] = num_stations;

    /* Create test visibilities and coordinates. */
    station_x = calloc(num_stations, DBL);
    station_y = calloc(num_stations, DBL);
    station_z = calloc(num_stations, DBL);
    uu        = calloc(num_times_baselines, DBL);
    vv        = calloc(num_times_baselines, DBL);
    ww        = calloc(num_times_baselines, DBL);
    vis_block = calloc(num_times_baselines * num_channels, DBL*8);
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
    for (i = 0; i < num_times_baselines * num_channels; ++i)
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
            sizeof(telescope_model_path), telescope_model_path, &status);

    /* Write number of tags per block. */
    oskar_binary_write_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_NUM_TAGS_PER_BLOCK, 0, num_tags_per_block,
            &status);

    /* Write dimensions. */
    oskar_binary_write_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_WRITE_AUTOCORRELATIONS, 0, 0, &status);
    oskar_binary_write_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_AMP_TYPE, 0, amp_type, &status);
    oskar_binary_write_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_MAX_TIMES_PER_BLOCK, 0,
            max_times_per_block, &status);
    oskar_binary_write_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_NUM_TIMES_TOTAL, 0, num_times_total, &status);
    oskar_binary_write_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_NUM_CHANNELS, 0, num_channels, &status);
    oskar_binary_write_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_NUM_STATIONS, 0, num_stations, &status);

    /* Write other visibility metadata. */
    oskar_binary_write_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_FREQ_START_HZ, 0, freq_start_inc[0], &status);
    oskar_binary_write_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_FREQ_INC_HZ, 0, freq_start_inc[1], &status);
    oskar_binary_write_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_CHANNEL_BANDWIDTH_HZ, 0,
            channel_bandwidth_hz, &status);
    oskar_binary_write_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_TIME_START_MJD_UTC, 0,
            time_start_inc[0], &status);
    oskar_binary_write_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_TIME_INC_SEC, 0, time_start_inc[1] * 86400.0,
            &status);
    oskar_binary_write_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_TIME_AVERAGE_SEC, 0,
            time_average_sec, &status);
    oskar_binary_write(h, OSKAR_DOUBLE, vis_header_group,
            OSKAR_VIS_HEADER_TAG_PHASE_CENTRE, 0,
            DBL*2, phase_centre, &status);
    oskar_binary_write(h, OSKAR_DOUBLE, vis_header_group,
            OSKAR_VIS_HEADER_TAG_TELESCOPE_CENTRE, 0,
            DBL*3, telescope_centre, &status);

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
        oskar_binary_write(h, OSKAR_DOUBLE, vis_block_group,
                OSKAR_VIS_BLOCK_TAG_FREQ_REF_INC_HZ, i,
                DBL*2, freq_start_inc, &status);
        oskar_binary_write(h, OSKAR_DOUBLE, vis_block_group,
                OSKAR_VIS_BLOCK_TAG_TIME_REF_INC_MJD_UTC, i,
                DBL*2, time_start_inc, &status);

        /* Write the visibility data. */
        oskar_binary_write(h, amp_type, vis_block_group,
                OSKAR_VIS_BLOCK_TAG_CROSS_CORRELATIONS, i,
                DBL*8 * num_times_baselines * num_channels,
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
    int i, precision, element_size, status = 0;
    oskar_Binary* h;
    const unsigned char vis_header_group = OSKAR_TAG_GROUP_VIS_HEADER;
    const unsigned char vis_block_group = OSKAR_TAG_GROUP_VIS_BLOCK;

    /* Data to read. */
    int amp_type, max_times_per_block, num_tags_per_block;
    int num_baselines, num_channels, num_stations, num_times_total;
    int num_blocks, num_times_baselines, dim_start_and_size[6];
    double phase_centre[2], telescope_centre[3];
    double freq_start_inc[2], time_start_inc[2];
    double channel_bandwidth_hz, time_average_sec;
    void *station_x, *station_y, *station_z, *vis_block, *uu, *vv, *ww;

    /* Open the test file for reading. */
    printf("Reading test file...\n");
    h = oskar_binary_create(filename, 'r', &status);

    /* Read number of tags per block. */
    oskar_binary_read_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_NUM_TAGS_PER_BLOCK, 0, &num_tags_per_block,
            &status);

    /* Read dimensions. */
    oskar_binary_read_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_AMP_TYPE, 0, &amp_type, &status);
    oskar_binary_read_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_MAX_TIMES_PER_BLOCK, 0,
            &max_times_per_block, &status);
    oskar_binary_read_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_NUM_TIMES_TOTAL, 0, &num_times_total, &status);
    oskar_binary_read_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_NUM_CHANNELS, 0, &num_channels, &status);
    oskar_binary_read_int(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_NUM_STATIONS, 0, &num_stations, &status);

    /* Read visibility metadata. */
    oskar_binary_read_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_FREQ_START_HZ, 0, &freq_start_inc[0], &status);
    oskar_binary_read_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_FREQ_INC_HZ, 0, &freq_start_inc[1], &status);
    oskar_binary_read_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_CHANNEL_BANDWIDTH_HZ, 0,
            &channel_bandwidth_hz, &status);
    oskar_binary_read_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_TIME_START_MJD_UTC, 0,
            &time_start_inc[0], &status);
    oskar_binary_read_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_TIME_INC_SEC, 0,
            &time_start_inc[1], &status);
    oskar_binary_read_double(h, vis_header_group,
            OSKAR_VIS_HEADER_TAG_TIME_AVERAGE_SEC, 0,
            &time_average_sec, &status);
    oskar_binary_read(h, OSKAR_DOUBLE, vis_header_group,
            OSKAR_VIS_HEADER_TAG_PHASE_CENTRE, 0,
            DBL*2, phase_centre, &status);
    oskar_binary_read(h, OSKAR_DOUBLE, vis_header_group,
            OSKAR_VIS_HEADER_TAG_TELESCOPE_CENTRE, 0,
            DBL*3, telescope_centre, &status);

    /* Get data type precision and element size from amplitude type. */
    precision = amp_type & 0x0F;
    element_size = (precision == OSKAR_DOUBLE ? DBL : FLT);

    /* Read the station coordinates. */
    station_x = calloc(num_stations, element_size);
    station_y = calloc(num_stations, element_size);
    station_z = calloc(num_stations, element_size);
    oskar_binary_read(h, precision, vis_header_group,
            OSKAR_VIS_HEADER_TAG_STATION_X_OFFSET_ECEF, 0,
            num_stations * element_size, station_x, &status);
    oskar_binary_read(h, precision, vis_header_group,
            OSKAR_VIS_HEADER_TAG_STATION_Y_OFFSET_ECEF, 0,
            num_stations * element_size, station_y, &status);
    oskar_binary_read(h, precision, vis_header_group,
            OSKAR_VIS_HEADER_TAG_STATION_Z_OFFSET_ECEF, 0,
            num_stations * element_size, station_z, &status);

    /* Print header data. */
    printf("Max. number of times per block: %d\n", max_times_per_block);
    printf("Total number of times: %d\n", num_times_total);
    printf("Number of stations: %d\n", num_stations);
    if (precision == OSKAR_DOUBLE)
    {
        for (i = 0; i < num_stations; ++i)
        {
            printf("Station[%d] at (%.3f, %.3f, %.3f)\n", i,
                    ((double*)station_x)[i],
                    ((double*)station_y)[i],
                    ((double*)station_z)[i]);
        }
    }
    else if (precision == OSKAR_SINGLE)
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
    uu        = calloc(num_times_baselines, element_size);
    vv        = calloc(num_times_baselines, element_size);
    ww        = calloc(num_times_baselines, element_size);
    vis_block = calloc(num_times_baselines * num_channels, 8*element_size);

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
        oskar_binary_read(h, OSKAR_DOUBLE, vis_block_group,
                OSKAR_VIS_BLOCK_TAG_FREQ_REF_INC_HZ, i,
                DBL*2, freq_start_inc, &status);
        oskar_binary_read(h, OSKAR_DOUBLE, vis_block_group,
                OSKAR_VIS_BLOCK_TAG_TIME_REF_INC_MJD_UTC, i,
                DBL*2, time_start_inc, &status);

        /* Get the number of times actually in the block. */
        start_time_idx      = dim_start_and_size[0];
        start_channel_idx   = dim_start_and_size[1];
        num_times           = dim_start_and_size[2];

        /* Read the visibility data. */
        oskar_binary_read(h, amp_type, vis_block_group,
                OSKAR_VIS_BLOCK_TAG_CROSS_CORRELATIONS, i,
                element_size*8 * num_times_baselines * num_channels,
                vis_block, &status);

        /* Read the baseline data. */
        oskar_binary_read(h, OSKAR_DOUBLE, vis_block_group,
                OSKAR_VIS_BLOCK_TAG_BASELINE_UU, i,
                element_size * num_times_baselines, uu, &status);
        oskar_binary_read(h, OSKAR_DOUBLE, vis_block_group,
                OSKAR_VIS_BLOCK_TAG_BASELINE_VV, i,
                element_size * num_times_baselines, vv, &status);
        oskar_binary_read(h, OSKAR_DOUBLE, vis_block_group,
                OSKAR_VIS_BLOCK_TAG_BASELINE_WW, i,
                element_size * num_times_baselines, ww, &status);

        /* Check for errors. */
        if (status) break;

        /* Print contents of the block. */
        for (t = 0; t < num_times; ++t)
        {
            double mjd_utc;

            /* Get the actual time of the sample. */
            mjd_utc = time_start_inc[0] +
                    time_start_inc[1] * (start_time_idx + t + 0.5);

            for (c = 0; c < num_channels; ++c)
            {
                double freq_hz;

                /* Get the actual frequency of the sample. */
                freq_hz = freq_start_inc[0] +
                        freq_start_inc[1] * (start_channel_idx + c);

                if (precision == OSKAR_DOUBLE)
                {
                    const double *u, *v, *w, *d;
                    u = (const double*) uu;
                    v = (const double*) vv;
                    w = (const double*) ww;
                    d = (const double*) vis_block;
                    for (b = 0; b < num_baselines; ++b)
                    {
                        int i, j;
                        i = 8 * (b + num_baselines * (c + num_channels * t));
                        j = b + num_baselines * t;
                        printf("Time %d (%.4f), Channel %d (%.3f MHz), "
                                "Baseline %d\n"
                                "    (U, V, W) = (%.3f, %.3f, %.3f)\n"
                                "            [%.3f, %.3f] [%.3f, %.3f]\n"
                                "            [%.3f, %.3f] [%.3f, %.3f]\n",
                                start_time_idx + t, mjd_utc,
                                start_channel_idx + c, freq_hz / 1e6, b,
                                u[j], v[j], w[j],
                                d[i + 0], d[i + 1], d[i + 2], d[i + 3],
                                d[i + 4], d[i + 5], d[i + 6], d[i + 7]);
                    }
                }
                else if (precision == OSKAR_SINGLE)
                {
                    const float *u, *v, *w, *d;
                    u = (const float*) uu;
                    v = (const float*) vv;
                    w = (const float*) ww;
                    d = (const float*) vis_block;
                    for (b = 0; b < num_baselines; ++b)
                    {
                        int i, j;
                        i = 8 * (b + num_baselines * (c + num_channels * t));
                        j = b + num_baselines * t;
                        printf("Time %d (%.4f), Channel %d (%.3f MHz), "
                                "Baseline %d\n"
                                "    (U, V, W) = (%.3f, %.3f, %.3f)\n"
                                "            [%.3f, %.3f] [%.3f, %.3f]\n"
                                "            [%.3f, %.3f] [%.3f, %.3f]\n",
                                start_time_idx + t, mjd_utc,
                                start_channel_idx + c, freq_hz / 1e6, b,
                                u[j], v[j], w[j],
                                d[i + 0], d[i + 1], d[i + 2], d[i + 3],
                                d[i + 4], d[i + 5], d[i + 6], d[i + 7]);
                    }
                }
            }
        }
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

int main(void)
{
    const char* filename = "test.vis";

    /* Write a test visibility data file. */
    write_test_vis(filename);

    /* Read the test visibility data file. */
    read_test_vis(filename);

    return 0;
}

