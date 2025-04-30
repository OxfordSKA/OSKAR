/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/oskar_cmath.h"
#include "telescope/oskar_telescope.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_evaluate_magnetic_field(
        oskar_Telescope* telescope,
        double year,
        int* status
)
{
    if (*status) return;
    int i = 0;
    const int num_stations = oskar_telescope_num_stations(telescope);
    for (i = 0; i < num_stations; ++i)
    {
        oskar_station_evaluate_magnetic_field(
                oskar_telescope_station(telescope, i), year, status
        );
    }
}

#ifdef __cplusplus
}
#endif
