/*
 * Copyright (c) 2016, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "telescope/oskar_telescope.h"
#include "math/oskar_cmath.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_log_summary(const oskar_Telescope* telescope,
        oskar_Log* log, int* status)
{
    if (*status) return;
    oskar_log_section(log, 'M', "Telescope model summary");
    oskar_log_value(log, 'M', 0, "Longitude [deg]", "%.3f",
            oskar_telescope_lon_rad(telescope) * 180.0 / M_PI);
    oskar_log_value(log, 'M', 0, "Latitude [deg]", "%.3f",
            oskar_telescope_lat_rad(telescope) * 180.0 / M_PI);
    oskar_log_value(log, 'M', 0, "Altitude [m]", "%.0f",
            oskar_telescope_alt_metres(telescope));
    oskar_log_value(log, 'M', 0, "Num. stations", "%d",
            oskar_telescope_num_stations(telescope));
    oskar_log_value(log, 'M', 0, "Max station size", "%d",
            oskar_telescope_max_station_size(telescope));
    oskar_log_value(log, 'M', 0, "Max station depth", "%d",
            oskar_telescope_max_station_depth(telescope));
    oskar_log_value(log, 'M', 0, "Identical stations", "%s",
            oskar_telescope_identical_stations(telescope) ? "true" : "false");
}

#ifdef __cplusplus
}
#endif
