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

#include <private_vis_block.h>
#include <oskar_vis_block.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_VisBlock* oskar_vis_block_create(int amp_type, int location,
        int num_times, int num_channels, int num_stations, int create_autocorr,
        int create_crosscorr, int* status)
{
    oskar_VisBlock* vis = 0;
    int num_autocorr = 0, num_xcorr = 0, num_baselines = 0, num_coords = 0;
    int *b_s1, *b_s2, i, s1, s2, type;

    /* Check all inputs. */
    if (!status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Check type. */
    if (oskar_type_is_double(amp_type))
        type = OSKAR_DOUBLE;
    else if (oskar_type_is_single(amp_type))
        type = OSKAR_SINGLE;
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return 0;
    }
    if (!oskar_type_is_complex(amp_type))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return 0;
    }

    /* Allocate the structure. */
    vis = (oskar_VisBlock*) malloc(sizeof(oskar_VisBlock));

    /* Set dimensions. */
    if (create_crosscorr)
        num_baselines = num_stations * (num_stations - 1) / 2;
    vis->dim_start_size[0] = 0; /* Global time index start of block. */
    vis->dim_start_size[1] = 0; /* Global frequency index start of block. */
    vis->dim_start_size[2] = num_times;
    vis->dim_start_size[3] = num_channels;
    vis->dim_start_size[4] = num_baselines;
    vis->dim_start_size[5] = num_stations;
    num_coords = num_times * num_baselines;
    num_xcorr  = num_times * num_baselines * num_channels;
    if (create_autocorr)
        num_autocorr = num_channels * num_times * num_stations;

    /* Create arrays. */
    vis->baseline_uu_metres = oskar_mem_create(type, location,
            num_coords, status);
    vis->baseline_vv_metres = oskar_mem_create(type, location,
            num_coords, status);
    vis->baseline_ww_metres = oskar_mem_create(type, location,
            num_coords, status);
    /*vis->baseline_num_channel_averages = oskar_mem_create(OSKAR_INT, location,
            num_baselines, status);*/
    /*vis->baseline_num_time_averages = oskar_mem_create(OSKAR_INT, location,
            num_baselines, status);*/
    vis->auto_correlations  = oskar_mem_create(amp_type, location,
            num_autocorr, status);
    vis->cross_correlations = oskar_mem_create(amp_type, location,
            num_xcorr, status);
    vis->a1 = oskar_mem_create(OSKAR_INT, OSKAR_CPU, num_baselines, status);
    vis->a2 = oskar_mem_create(OSKAR_INT, OSKAR_CPU, num_baselines, status);

    /* TODO Move these to oskar_MeasurementSet. */
    /* Evaluate baseline index arrays for Measurement Set export. */
    b_s1 = oskar_mem_int(vis->a1, status);
    b_s2 = oskar_mem_int(vis->a2, status);
    for (s1 = 0, i = 0; s1 < num_stations; ++s1)
    {
        for (s2 = create_autocorr ? s1 : s1 + 1; s2 < num_stations; ++i, ++s2)
        {
            b_s1[i] = s1;
            b_s2[i] = s2;
        }
    }

    /* Clear index arrays. */
    /*oskar_mem_clear_contents(vis->baseline_num_channel_averages, status);*/
    /*oskar_mem_clear_contents(vis->baseline_num_time_averages, status);*/

    /* Return handle to structure. */
    return vis;
}

#ifdef __cplusplus
}
#endif
