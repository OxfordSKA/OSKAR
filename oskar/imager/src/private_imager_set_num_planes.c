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

#include "imager/private_imager.h"
#include "imager/private_imager_set_num_planes.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SEC2DAYS 1.15740740740740740740741e-5

void oskar_imager_set_num_planes(oskar_Imager* h)
{
    if (h->num_planes > 0) return;

    /* Set image meta-data. */
    h->im_num_channels = (h->chan_snaps ?
            1 + h->vis_chan_range[1] - h->vis_chan_range[0] : 1);
    h->im_num_times = (h->time_snaps ?
            1 + h->vis_time_range[1] - h->vis_time_range[0] : 1);
    h->im_time_start_mjd_utc = h->vis_time_start_mjd_utc +
            (h->vis_time_range[0] * h->time_inc_sec * SEC2DAYS);
    if (h->chan_snaps)
    {
        h->im_freq_start_hz = h->vis_freq_start_hz +
                h->vis_chan_range[0] * h->freq_inc_hz;
    }
    else
    {
        double chan0 = 0.5 * (h->vis_chan_range[1] - h->vis_chan_range[0]);
        h->im_freq_start_hz = h->vis_freq_start_hz + chan0 * h->freq_inc_hz;
    }
    h->num_planes = h->im_num_times * h->im_num_channels * h->im_num_pols;
}


#ifdef __cplusplus
}
#endif
