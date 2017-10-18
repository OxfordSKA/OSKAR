/*
 * Copyright (c) 2012-2016, The University of Oxford
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
#include "utility/oskar_dir.h"
#include "math/oskar_cmath.h"

#ifdef __cplusplus
extern "C" {
#endif

static void oskar_telescope_save_private(const oskar_Telescope* telescope,
        const char* dir_path, const oskar_Station* station, int depth,
        int* status);

void oskar_telescope_save(const oskar_Telescope* telescope,
        const char* dir_path, int* status)
{
    oskar_telescope_save_private(telescope, dir_path, NULL, 0, status);
}

static void oskar_telescope_save_private(const oskar_Telescope* telescope,
        const char* dir_path, const oskar_Station* station, int depth,
        int* status)
{
    char* path;
    int i = 0, num_stations = 0;

    if (depth == 0)
    {
        /* Check if directory already exists, and remove it if so. */
        if (oskar_dir_exists(dir_path))
        {
            if (!oskar_dir_remove(dir_path))
            {
                *status = OSKAR_ERR_FILE_IO;
                return;
            }
        }
    }

    /* Create the directory if it doesn't exist. */
    if (!oskar_dir_exists(dir_path))
        oskar_dir_mkpath(dir_path);

    if (depth == 0)
    {
        /* Write the reference position. */
        FILE* file;
        path = oskar_dir_get_path(dir_path, "position.txt");
        file = fopen(path, "w");
        free(path);
        if (!file)
        {
            *status = OSKAR_ERR_FILE_IO;
            return;
        }
        fprintf(file, "%.12f, %.12f, %.12f\n",
                oskar_telescope_lon_rad(telescope) * 180.0 / M_PI,
                oskar_telescope_lat_rad(telescope) * 180.0 / M_PI,
                oskar_telescope_alt_metres(telescope));
        fclose(file);

        /* Write the station coordinates. */
        path = oskar_dir_get_path(dir_path, "layout.txt");
        oskar_telescope_save_layout(telescope, path, status);
        free(path);

        /* Get the number of stations. */
        num_stations = oskar_telescope_num_stations(telescope);
    }
    else
    {
        /* Write the station configuration data. */
        path = oskar_dir_get_path(dir_path, "layout.txt");
        oskar_station_save_layout(path, station, status);
        free(path);
        path = oskar_dir_get_path(dir_path, "mount_types.txt");
        oskar_station_save_mount_types(path, station, status);
        free(path);
        if (!oskar_station_has_child(station))
        {
            path = oskar_dir_get_path(dir_path, "feed_angle_x.txt");
            oskar_station_save_feed_angle(path, station, 1, status);
            free(path);
            path = oskar_dir_get_path(dir_path, "feed_angle_y.txt");
            oskar_station_save_feed_angle(path, station, 0, status);
            free(path);
        }
        if (oskar_station_apply_element_errors(station))
        {
            path = oskar_dir_get_path(dir_path, "gain_phase.txt");
            oskar_station_save_gain_phase(path, station, status);
            free(path);
        }
        if (oskar_station_apply_element_weight(station))
        {
            path = oskar_dir_get_path(dir_path, "apodisation.txt");
            oskar_station_save_apodisation(path, station, status);
            free(path);
        }
        if (oskar_station_num_element_types(station) > 1)
        {
            path = oskar_dir_get_path(dir_path, "element_types.txt");
            oskar_station_save_element_types(path, station, status);
            free(path);
        }
        if (oskar_station_num_permitted_beams(station) > 0)
        {
            path = oskar_dir_get_path(dir_path, "permitted_beams.txt");
            oskar_station_save_permitted_beams(path, station, status);
            free(path);
        }

        /* Get the number of stations. */
        if (oskar_station_has_child(station))
            num_stations = oskar_station_num_elements(station);
    }

    /* Recursive call to write stations. */
    for (i = 0; i < num_stations; ++i)
    {
        /* Get station name, and a pointer to the station to save. */
        const oskar_Station* s;
        char station_name[128], *path;
        sprintf(station_name, "level%1d_%03d", depth, i);
        s = (depth == 0) ? oskar_telescope_station_const(telescope, i) :
                oskar_station_child_const(station, i);

        /* Save this station. */
        path = oskar_dir_get_path(dir_path, station_name);
        oskar_telescope_save_private(telescope, path, s, depth + 1, status);
        free(path);
    }
}

#ifdef __cplusplus
}
#endif
