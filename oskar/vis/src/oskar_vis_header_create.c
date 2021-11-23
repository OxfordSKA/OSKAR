/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "vis/private_vis_header.h"
#include "vis/oskar_vis_header.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_VisHeader* oskar_vis_header_create(int amp_type, int coord_precision,
        int max_times_per_block, int num_times_total,
        int max_channels_per_block, int num_channels_total, int num_stations,
        int write_autocorr, int write_crosscor, int* status)
{
    int i = 0, j = 0;
    oskar_VisHeader* hdr = 0;
    if (*status) return 0;

    /* Check type. */
    if (!oskar_type_is_complex(amp_type))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return 0;
    }

    /* Allocate the structure. */
    hdr = (oskar_VisHeader*) calloc(1, sizeof(oskar_VisHeader));
    if (!hdr)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        return 0;
    }
    hdr->amp_type = amp_type;
    hdr->coord_precision = coord_precision;

    /* Set number of tags per block in the binary file. */
    /* This must be updated if the number of fields written to file from
     * the oskar_VisBlock structure is changed. */
    hdr->num_tags_per_block = 1;
    if (write_crosscor) hdr->num_tags_per_block += 4;
    if (write_autocorr) hdr->num_tags_per_block += 1;

    /* Set dimensions. */
    if (max_channels_per_block <= 0)
    {
        max_channels_per_block = num_channels_total;
    }
    if (max_times_per_block <= 0)
    {
        max_times_per_block = num_times_total;
    }
    hdr->max_times_per_block    = max_times_per_block;
    hdr->num_times_total        = num_times_total;
    hdr->max_channels_per_block = max_channels_per_block;
    hdr->num_channels_total     = num_channels_total;
    hdr->num_stations           = num_stations;
    hdr->write_autocorr         = write_autocorr;
    hdr->write_crosscorr        = write_crosscor;

    /* Set default polarisation type. */
    if (oskar_type_is_matrix(amp_type))
    {
        oskar_vis_header_set_pol_type(hdr,
                OSKAR_VIS_POL_TYPE_LINEAR_XX_XY_YX_YY, status);
    }
    else
    {
        oskar_vis_header_set_pol_type(hdr,
                OSKAR_VIS_POL_TYPE_STOKES_I, status);
    }

    /* Initialise arrays. */
    hdr->telescope_path = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 0, status);
    hdr->settings = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 0, status);
    for (i = 0; i < 3; ++i)
    {
        hdr->station_offset_ecef_metres[i] = oskar_mem_create(coord_precision,
                OSKAR_CPU, num_stations, status);
        hdr->element_enu_metres[i] = (oskar_Mem**) calloc(
                num_stations, sizeof(oskar_Mem*));
        for (j = 0; j < num_stations; ++j)
        {
            hdr->element_enu_metres[i][j] = oskar_mem_create(coord_precision,
                    OSKAR_CPU, 0, status);
        }
    }

    return hdr;
}

#ifdef __cplusplus
}
#endif
