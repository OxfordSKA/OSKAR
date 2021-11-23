/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/private_telescope.h"
#include "telescope/oskar_telescope.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_set_station_coords(oskar_Telescope* dst, int index,
        const double true_geodetic[3],
        const double measured_offset_ecef[3], const double true_offset_ecef[3],
        const double measured_enu[3], const double true_enu[3], int* status)
{
    int i = 0;
    for (i = 0; i < 3; ++i)
    {
        oskar_mem_set_element_real(dst->station_true_geodetic_rad[i],
                index, true_geodetic[i], status);
        oskar_mem_set_element_real(dst->station_measured_offset_ecef_metres[i],
                index, measured_offset_ecef[i], status);
        oskar_mem_set_element_real(dst->station_true_offset_ecef_metres[i],
                index, true_offset_ecef[i], status);
        oskar_mem_set_element_real(dst->station_measured_enu_metres[i],
                index, measured_enu[i], status);
        oskar_mem_set_element_real(dst->station_true_enu_metres[i],
                index, true_enu[i], status);
    }
}

#ifdef __cplusplus
}
#endif
