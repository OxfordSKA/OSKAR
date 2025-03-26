/*
 * Copyright (c) 2015-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "vis/private_vis_header.h"
#include "vis/oskar_vis_header.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_vis_header_free(oskar_VisHeader* hdr, int* status)
{
    int i = 0, j = 0, k = 0;
    if (!hdr) return;
    oskar_mem_free(hdr->telescope_path, status);
    oskar_mem_free(hdr->settings, status);
    oskar_mem_free(hdr->station_diameter_m, status);
    for (i = 0; i < hdr->num_stations; ++i)
    {
        oskar_mem_free(hdr->station_name[i], status);
    }
    free(hdr->station_name);
    for (i = 0; i < 3; ++i)
    {
        oskar_mem_free(hdr->station_offset_ecef_metres[i], status);
        for (j = 0; j < hdr->num_stations; ++j)
        {
            oskar_mem_free(hdr->element_enu_metres[i][j], status);
        }
        free(hdr->element_enu_metres[i]);
    }
    for (i = 0; i < 2; ++i)
    {
        for (j = 0; j < 3; ++j)
        {
            for (k = 0; k < hdr->num_stations; ++k)
            {
                oskar_mem_free(hdr->element_euler_angle_rad[i][j][k], status);
            }
            free(hdr->element_euler_angle_rad[i][j]);
        }
    }
    free(hdr);
}

#ifdef __cplusplus
}
#endif
