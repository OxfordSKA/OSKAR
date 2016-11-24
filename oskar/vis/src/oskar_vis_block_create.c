/*
 * Copyright (c) 2015-2016, The University of Oxford
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
#include "vis/oskar_vis_block.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_VisBlock* oskar_vis_block_create(int location, int amp_type,
        int num_times, int num_channels, int num_stations,
        int create_crosscorr, int create_autocorr, int* status)
{
    oskar_VisBlock* vis = 0;
    int type;

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
    vis = (oskar_VisBlock*) calloc(1, sizeof(oskar_VisBlock));
    if (!vis)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        return 0;
    }

    /* Store metadata. */
    vis->has_auto_correlations = create_autocorr;
    vis->has_cross_correlations = create_crosscorr;

    /* Create arrays. */
    vis->baseline_uu_metres = oskar_mem_create(type, location, 0, status);
    vis->baseline_vv_metres = oskar_mem_create(type, location, 0, status);
    vis->baseline_ww_metres = oskar_mem_create(type, location, 0, status);
    vis->auto_correlations  = oskar_mem_create(amp_type, location, 0, status);
    vis->cross_correlations = oskar_mem_create(amp_type, location, 0, status);

    /* Set dimensions. */
    oskar_vis_block_resize(vis, num_times, num_channels, num_stations, status);

    /* Return handle to structure. */
    return vis;
}

#ifdef __cplusplus
}
#endif
