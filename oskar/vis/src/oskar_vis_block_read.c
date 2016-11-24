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

#include "vis/private_vis_block.h"
#include "binary/oskar_binary.h"
#include "mem/oskar_binary_read_mem.h"
#include "vis/oskar_vis_block.h"
#include "vis/oskar_vis_header.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_vis_block_read(oskar_VisBlock* vis, const oskar_VisHeader* hdr,
        oskar_Binary* h, int block_index, int* status)
{
    int num_tags_per_block;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Set query start index. */
    num_tags_per_block = oskar_vis_header_num_tags_per_block(hdr);
    oskar_binary_set_query_search_start(h, block_index * num_tags_per_block,
            status);

    /* Read visibility metadata. */
    oskar_binary_read(h, OSKAR_INT,
            OSKAR_TAG_GROUP_VIS_BLOCK,
            OSKAR_VIS_BLOCK_TAG_DIM_START_AND_SIZE, block_index,
            sizeof(int) * 6, vis->dim_start_size, status);

    /* Read the auto-correlation data. */
    if (oskar_vis_header_write_auto_correlations(hdr))
    {
        oskar_binary_read_mem(h, vis->auto_correlations,
                OSKAR_TAG_GROUP_VIS_BLOCK,
                OSKAR_VIS_BLOCK_TAG_AUTO_CORRELATIONS, block_index, status);
    }

    /* Read the cross-correlation data. */
    if (oskar_vis_header_write_cross_correlations(hdr))
    {
        oskar_binary_read_mem(h, vis->cross_correlations,
                OSKAR_TAG_GROUP_VIS_BLOCK,
                OSKAR_VIS_BLOCK_TAG_CROSS_CORRELATIONS, block_index, status);

        /* Read the baseline coordinate data. */
        oskar_binary_read_mem(h, vis->baseline_uu_metres,
                OSKAR_TAG_GROUP_VIS_BLOCK,
                OSKAR_VIS_BLOCK_TAG_BASELINE_UU, block_index, status);
        oskar_binary_read_mem(h, vis->baseline_vv_metres,
                OSKAR_TAG_GROUP_VIS_BLOCK,
                OSKAR_VIS_BLOCK_TAG_BASELINE_VV, block_index, status);
        oskar_binary_read_mem(h, vis->baseline_ww_metres,
                OSKAR_TAG_GROUP_VIS_BLOCK,
                OSKAR_VIS_BLOCK_TAG_BASELINE_WW, block_index, status);
    }
}

#ifdef __cplusplus
}
#endif
