/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/private_imager.h"
#include "imager/private_imager_set_num_planes.h"
#include "log/oskar_log.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_set_num_planes(oskar_Imager* h, int* status)
{
    int i = 0;
    if (*status || h->num_planes > 0) return;
    if (h->num_sel_freqs == 0)
    {
        oskar_log_error(h->log,
                "Input visibility channel frequencies not set.");
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return;
    }
    else
    {
        oskar_log_message(h->log, 'M', 0, "Using %d frequency channel(s)",
                h->num_sel_freqs);
        oskar_log_message(h->log, 'M', 1, "Range %.3f MHz to %.3f MHz",
                h->sel_freqs[0] * 1e-6,
                h->sel_freqs[h->num_sel_freqs - 1] * 1e-6);
    }

    /* Set image meta-data. */
    h->num_im_channels = h->chan_snaps ? h->num_sel_freqs : 1;
    h->im_freqs = (double*) realloc(h->im_freqs,
            h->num_im_channels * sizeof(double));
    if (h->chan_snaps)
    {
        for (i = 0; i < h->num_sel_freqs; ++i)
        {
            h->im_freqs[i] = h->sel_freqs[i];
        }
    }
    else
    {
        h->im_freqs[0] = 0.0;
        for (i = 0; i < h->num_sel_freqs; ++i)
        {
            h->im_freqs[0] += h->sel_freqs[i];
        }
        h->im_freqs[0] /= h->num_sel_freqs;
    }
    h->num_planes = h->num_im_channels * h->num_im_pols;
}

#ifdef __cplusplus
}
#endif
