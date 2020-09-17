/*
 * Copyright (c) 2015-2020, The OSKAR Developers.
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
    oskar_VisHeader* hdr = 0;

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* Check type. */
    if (!oskar_type_is_complex(amp_type))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return 0;
    }

    /* Allocate the structure. */
    hdr = (oskar_VisHeader*) malloc(sizeof(oskar_VisHeader));
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
        max_channels_per_block = num_channels_total;
    if (max_times_per_block <= 0)
        max_times_per_block = num_times_total;
    hdr->max_times_per_block    = max_times_per_block;
    hdr->num_times_total        = num_times_total;
    hdr->max_channels_per_block = max_channels_per_block;
    hdr->num_channels_total     = num_channels_total;
    hdr->num_stations           = num_stations;

    /* Set default polarisation type. */
    if (oskar_type_is_matrix(amp_type))
        hdr->pol_type = OSKAR_VIS_POL_TYPE_LINEAR_XX_XY_YX_YY;
    else
        hdr->pol_type = OSKAR_VIS_POL_TYPE_STOKES_I;

    /* Initialise meta-data. */
    hdr->write_autocorr = write_autocorr;
    hdr->write_crosscorr = write_crosscor;
    hdr->freq_start_hz = 0.0;
    hdr->freq_inc_hz = 0.0;
    hdr->channel_bandwidth_hz = 0.0;
    hdr->time_start_mjd_utc = 0.0;
    hdr->time_inc_sec = 0.0;
    hdr->time_average_sec = 0.0;
    hdr->phase_centre_type = 0;
    hdr->phase_centre_deg[0] = 0.0;
    hdr->phase_centre_deg[1] = 0.0;
    hdr->telescope_centre_lon_deg = 0.0;
    hdr->telescope_centre_lat_deg = 0.0;
    hdr->telescope_centre_alt_m = 0.0;

    /* Initialise CPU memory. */
    hdr->telescope_path = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 0, status);
    hdr->settings = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 0, status);
    hdr->station_x_offset_ecef_metres = oskar_mem_create(coord_precision,
            OSKAR_CPU, num_stations, status);
    hdr->station_y_offset_ecef_metres = oskar_mem_create(coord_precision,
            OSKAR_CPU, num_stations, status);
    hdr->station_z_offset_ecef_metres = oskar_mem_create(coord_precision,
            OSKAR_CPU, num_stations, status);

    /* Return handle to structure. */
    return hdr;
}

#ifdef __cplusplus
}
#endif
