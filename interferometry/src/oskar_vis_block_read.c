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
#include <oskar_binary.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_vis_block_read(oskar_VisBlock* vis, oskar_Binary* h,
        int block_index, int* status)
{
    /* Check all inputs. */
    if (!vis || !h || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Set query start index. */
    oskar_binary_set_query_search_start(h, block_index * 9, status);

    /* Read visibility metadata. */
    oskar_binary_read(h, OSKAR_INT,
            OSKAR_TAG_GROUP_VIS_BLOCK,
            OSKAR_VIS_BLOCK_TAG_DIM_SIZE, block_index,
            sizeof(int) * 4, vis->dim_size, status);
    oskar_binary_read(h, OSKAR_DOUBLE,
            OSKAR_TAG_GROUP_VIS_BLOCK,
            OSKAR_VIS_BLOCK_TAG_FREQ_RANGE_HZ, block_index,
            sizeof(double) * 2, vis->freq_range_hz, status);
    oskar_binary_read(h, OSKAR_DOUBLE,
            OSKAR_TAG_GROUP_VIS_BLOCK,
            OSKAR_VIS_BLOCK_TAG_TIME_RANGE_MJD_UTC_SEC, block_index,
            sizeof(double) * 2, vis->time_range_mjd_utc_sec, status);

    /* Read the baseline data. */
    oskar_binary_read_mem(h, vis->baseline_uu_metres,
            OSKAR_TAG_GROUP_VIS_BLOCK,
            OSKAR_VIS_BLOCK_TAG_BASELINE_UU, block_index, status);
    oskar_binary_read_mem(h, vis->baseline_vv_metres,
            OSKAR_TAG_GROUP_VIS_BLOCK,
            OSKAR_VIS_BLOCK_TAG_BASELINE_VV, block_index, status);
    oskar_binary_read_mem(h, vis->baseline_ww_metres,
            OSKAR_TAG_GROUP_VIS_BLOCK,
            OSKAR_VIS_BLOCK_TAG_BASELINE_WW, block_index, status);
    oskar_binary_read_mem(h, vis->baseline_num_channel_averages,
            OSKAR_TAG_GROUP_VIS_BLOCK,
            OSKAR_VIS_BLOCK_TAG_BASELINE_NUM_CHANNEL_AVERAGES,
            block_index, status);
    oskar_binary_read_mem(h, vis->baseline_num_time_averages,
            OSKAR_TAG_GROUP_VIS_BLOCK,
            OSKAR_VIS_BLOCK_TAG_BASELINE_NUM_TIME_AVERAGES,
            block_index, status);

    /* Read the visibility data. */
    oskar_binary_read_mem(h, vis->amplitude,
            OSKAR_TAG_GROUP_VIS_BLOCK,
            OSKAR_VIS_BLOCK_TAG_AMPLITUDE, block_index, status);
}

#ifdef __cplusplus
}
#endif
