/*
 * Copyright (c) 2011-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/private_telescope.h"
#include "telescope/oskar_telescope.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_free(oskar_Telescope* telescope, int* status)
{
    int i = 0;
    if (!telescope) return;

    /* Free the arrays. */
    for (i = 0; i < 3; ++i)
    {
        oskar_mem_free(telescope->station_true_geodetic_rad[i], status);
        oskar_mem_free(telescope->station_true_offset_ecef_metres[i], status);
        oskar_mem_free(telescope->station_true_enu_metres[i], status);
        oskar_mem_free(telescope->station_measured_offset_ecef_metres[i], status);
        oskar_mem_free(telescope->station_measured_enu_metres[i], status);
    }
    oskar_mem_free(telescope->station_type_map, status);
    oskar_mem_free(telescope->tec_screen_path, status);

    /* Free the gain model. */
    oskar_gains_free(telescope->gains, status);

    /* Free the HARP data. */
    oskar_mem_free(telescope->harp_freq_cpu, status);
    for (i = 0; i < telescope->harp_num_freq; ++i)
    {
        oskar_harp_free(telescope->harp_data[i]);
    }
    free(telescope->harp_data);

    /* Free each station. */
    for (i = 0; i < telescope->num_station_models; ++i)
    {
        oskar_station_free(oskar_telescope_station(telescope, i), status);
    }

    /* Free the station array. */
    free(telescope->station);

    /* Free the structure itself. */
    free(telescope);
}

#ifdef __cplusplus
}
#endif
