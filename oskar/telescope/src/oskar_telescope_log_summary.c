/*
 * Copyright (c) 2016-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "log/oskar_log.h"
#include "math/oskar_cmath.h"
#include "telescope/oskar_telescope.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_log_summary(const oskar_Telescope* telescope,
        oskar_Log* log, const int* status)
{
    if (*status) return;
    oskar_log_section(log, 'M', "Telescope model summary");
    oskar_log_value(log, 'M', 0, "Longitude [deg]", "%.3f",
            oskar_telescope_lon_rad(telescope) * 180.0 / M_PI);
    oskar_log_value(log, 'M', 0, "Latitude [deg]", "%.3f",
            oskar_telescope_lat_rad(telescope) * 180.0 / M_PI);
    oskar_log_value(log, 'M', 0, "Altitude [m]", "%.0f",
            oskar_telescope_alt_metres(telescope));
    oskar_log_value(log, 'M', 0, "Number of stations", "%d",
            oskar_telescope_num_stations(telescope));
    oskar_log_value(log, 'M', 0, "Number of station models", "%d",
            oskar_telescope_num_station_models(telescope));
    oskar_log_value(log, 'M', 0, "Max station size", "%d",
            oskar_telescope_max_station_size(telescope));
    oskar_log_value(log, 'M', 0, "Max station depth", "%d",
            oskar_telescope_max_station_depth(telescope));
}

#ifdef __cplusplus
}
#endif
