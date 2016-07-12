/*
 * Copyright (c) 2016, The University of Oxford
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

void oskar_vis_block_resize(oskar_VisBlock* vis, int num_times,
        int num_channels, int num_stations, int* status)
{
    int num_autocorr = 0, num_xcorr = 0, num_baselines = 0, num_coords = 0;
    int old_num_stations;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the old number of stations. */
    old_num_stations  = vis->dim_start_size[5];

    /* Set dimensions. */
    if (vis->has_cross_correlations)
        num_baselines = num_stations * (num_stations - 1) / 2;
    vis->dim_start_size[2] = num_times;
    vis->dim_start_size[3] = num_channels;
    vis->dim_start_size[4] = num_baselines;
    vis->dim_start_size[5] = num_stations;
    num_coords = num_times * num_baselines;
    num_xcorr  = num_times * num_baselines * num_channels;
    if (vis->has_auto_correlations)
        num_autocorr = num_channels * num_times * num_stations;

    /* Resize arrays as required. */
    oskar_mem_realloc(vis->baseline_uu_metres, num_coords, status);
    oskar_mem_realloc(vis->baseline_vv_metres, num_coords, status);
    oskar_mem_realloc(vis->baseline_ww_metres, num_coords, status);
    oskar_mem_realloc(vis->auto_correlations, num_autocorr, status);
    oskar_mem_realloc(vis->cross_correlations, num_xcorr, status);
    oskar_mem_realloc(vis->a1, num_baselines + num_stations, status);
    oskar_mem_realloc(vis->a2, num_baselines + num_stations, status);

    /* Re-evaluate baseline index arrays for Measurement Set if required. */
    if (num_stations != old_num_stations &&
            (vis->has_cross_correlations || vis->has_auto_correlations))
    {
        int *b_s1, *b_s2, i, s1, s2;
        b_s1 = oskar_mem_int(vis->a1, status);
        b_s2 = oskar_mem_int(vis->a2, status);
        for (s1 = 0, i = 0; s1 < num_stations; ++s1)
        {
            if (vis->has_auto_correlations)
            {
                b_s1[i] = s1;
                b_s2[i] = s1;
                ++i;
            }
            if (vis->has_cross_correlations)
            {
                for (s2 = s1 + 1; s2 < num_stations; ++i, ++s2)
                {
                    b_s1[i] = s1;
                    b_s2[i] = s2;
                }
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
