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

#ifdef __cplusplus
extern "C" {
#endif

oskar_VisBlock* oskar_vis_block_create_from_header(int location,
        const oskar_VisHeader* hdr, int* status)
{
    oskar_VisBlock* vis = 0;
    int amp_type = 0, num_times = 0, num_channels = 0, num_stations = 0;
    int create_crosscorr = 0, create_autocorr = 0;

    /* Get values from header. */
    amp_type         = oskar_vis_header_amp_type(hdr);
    num_times        = oskar_vis_header_max_times_per_block(hdr);
    num_channels     = oskar_vis_header_max_channels_per_block(hdr);
    num_stations     = oskar_vis_header_num_stations(hdr);
    create_autocorr  = oskar_vis_header_write_auto_correlations(hdr);
    create_crosscorr = oskar_vis_header_write_cross_correlations(hdr);

    vis = oskar_vis_block_create(location, amp_type, num_times, num_channels,
            num_stations, create_crosscorr, create_autocorr, status);

    /* Return handle to structure. */
    return vis;
}

#ifdef __cplusplus
}
#endif
