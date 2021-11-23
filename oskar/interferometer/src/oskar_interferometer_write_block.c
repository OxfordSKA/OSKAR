/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "interferometer/private_interferometer.h"
#include "interferometer/oskar_interferometer.h"
#include "vis/oskar_vis_block_write_ms.h"
#include "vis/oskar_vis_header_write_ms.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_interferometer_write_block(oskar_Interferometer* h,
        const oskar_VisBlock* block, int block_index, int* status)
{
    if (*status) return;
    oskar_timer_resume(h->tmr_write);
#ifndef OSKAR_NO_MS
    if (h->ms_name && !h->ms)
    {
        h->ms = oskar_vis_header_write_ms(h->header, h->ms_name,
                h->force_polarised_ms, status);
    }
    if (h->ms) oskar_vis_block_write_ms(block, h->header, h->ms, status);
#endif
    if (h->vis_name && !h->vis)
    {
        h->vis = oskar_vis_header_write(h->header, h->vis_name, status);
    }
    if (h->vis) oskar_vis_block_write(block, h->vis, block_index, status);
    oskar_timer_pause(h->tmr_write);
}

#ifdef __cplusplus
}
#endif
