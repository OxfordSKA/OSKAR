/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "vis/private_vis_header.h"
#include "vis/oskar_vis_header.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_vis_header_free(oskar_VisHeader* hdr, int* status)
{
    int i = 0, j = 0;
    if (!hdr) return;
    oskar_mem_free(hdr->telescope_path, status);
    oskar_mem_free(hdr->settings, status);
    for (i = 0; i < 3; ++i)
    {
        oskar_mem_free(hdr->station_offset_ecef_metres[i], status);
        for (j = 0; j < hdr->num_stations; ++j)
        {
            oskar_mem_free(hdr->element_enu_metres[i][j], status);
        }
        free(hdr->element_enu_metres[i]);
    }
    free(hdr);
}

#ifdef __cplusplus
}
#endif
