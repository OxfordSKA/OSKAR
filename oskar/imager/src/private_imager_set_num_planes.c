/*
 * Copyright (c) 2016-2017, The University of Oxford
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

void oskar_imager_set_num_planes(oskar_Imager* h, int* status)
{
    int i;
    if (*status || h->num_planes > 0) return;
    if (h->num_sel_freqs == 0)
    {
        oskar_log_error(h->log,
                "Input visibility channel frequencies not set.");
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return;
    }

    /* Set image meta-data. */
    h->num_im_channels = h->chan_snaps ? h->num_sel_freqs : 1;
    h->im_freqs = (double*) realloc(h->im_freqs,
            h->num_im_channels * sizeof(double));
    if (h->chan_snaps)
    {
        for (i = 0; i < h->num_sel_freqs; ++i)
            h->im_freqs[i] = h->sel_freqs[i];
    }
    else
    {
        h->im_freqs[0] = 0.0;
        for (i = 0; i < h->num_sel_freqs; ++i)
            h->im_freqs[0] += h->sel_freqs[i];
        h->im_freqs[0] /= h->num_sel_freqs;
    }
    h->num_planes = h->num_im_channels * h->num_im_pols;
}


#ifdef __cplusplus
}
#endif
