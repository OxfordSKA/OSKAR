/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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
    char* path = 0;
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
    {
        oskar_dir_mkpath(dir_path);
    }

    if (depth == 0)
    {
        /* Write the reference position. */
        FILE* file = 0;
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

        /* Write the station type mapping. */
        path = oskar_dir_get_path(dir_path, "station_type_map.txt");
        file = fopen(path, "w");
        free(path);
        oskar_mem_save_ascii(file, 1, 0,
                oskar_telescope_num_stations(telescope), status,
                oskar_telescope_station_type_map_const(telescope));
        fclose(file);

        /* Get the number of stations. */
        num_stations = oskar_telescope_num_station_models(telescope);
    }
    else
    {
        int num_feeds = 2, name_offset = 1;

        /* Write the station configuration data. */
        path = oskar_dir_get_path(dir_path, "mount_types.txt");
        oskar_station_save_mount_types(station, path, status);
        free(path);
        if (!oskar_station_has_child(station))
        {
            path = oskar_dir_get_path(dir_path, "feed_angle_x.txt");
            oskar_station_save_feed_angle(station, 0, path, status);
            free(path);
            path = oskar_dir_get_path(dir_path, "feed_angle_y.txt");
            oskar_station_save_feed_angle(station, 1, path, status);
            free(path);
        }
        if (oskar_station_num_element_types(station) > 1)
        {
            path = oskar_dir_get_path(dir_path, "element_types.txt");
            oskar_station_save_element_types(station, path, status);
            free(path);
        }
        if (oskar_station_num_permitted_beams(station) > 0)
        {
            path = oskar_dir_get_path(dir_path, "permitted_beams.txt");
            oskar_station_save_permitted_beams(station, path, status);
            free(path);
        }
        if (oskar_station_common_pol_beams(station))
        {
            num_feeds = 1;
            name_offset = 0;
        }
        const char* layout_name[] = {
                "layout.txt",
                "layout_x.txt",
                "layout_y.txt"};
        const char* cable_name[] = {
                "cable_length_error.txt",
                "cable_length_error_x.txt",
                "cable_length_error_y.txt"};
        const char* gain_phase_name[] = {
                "gain_phase.txt",
                "gain_phase_x.txt",
                "gain_phase_y.txt"};
        const char* apodisation_name[] = {
                "apodisation.txt",
                "apodisation_x.txt",
                "apodisation_y.txt"};
        for (i = 0; i < num_feeds; ++i)
        {
            const int i_name = i + name_offset;
            path = oskar_dir_get_path(dir_path, layout_name[i_name]);
            oskar_station_save_layout(station, i, path, status);
            free(path);
            path = oskar_dir_get_path(dir_path, cable_name[i_name]);
            oskar_station_save_cable_length_error(station, i, path, status);
            free(path);
            if (oskar_station_apply_element_errors(station))
            {
                path = oskar_dir_get_path(dir_path, gain_phase_name[i_name]);
                oskar_station_save_gain_phase(station, i, path, status);
                free(path);
            }
            if (oskar_station_apply_element_weight(station))
            {
                path = oskar_dir_get_path(dir_path, apodisation_name[i_name]);
                oskar_station_save_apodisation(station, i, path, status);
                free(path);
            }
        }

        /* Get the number of stations. */
        if (oskar_station_has_child(station))
        {
            num_stations = oskar_station_num_elements(station);
        }
    }

    /* Recursive call to write stations. */
    for (i = 0; i < num_stations; ++i)
    {
        /* Get station name, and a pointer to the station to save. */
        const oskar_Station* s = 0;
        char station_name[128], *path = 0;
        sprintf(station_name, "level%1d_%03d", depth, i);
        s = (depth == 0) ? oskar_telescope_station_const(telescope, i) :
                oskar_station_child_const(station, i);

        /* Save this station. */
        if (!s) continue;
        path = oskar_dir_get_path(dir_path, station_name);
        oskar_telescope_save_private(telescope, path, s, depth + 1, status);
        free(path);
    }
}

#ifdef __cplusplus
}
#endif
