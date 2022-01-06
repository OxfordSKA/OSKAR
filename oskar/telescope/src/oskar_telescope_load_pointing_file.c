/*
 * Copyright (c) 2013-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/private_telescope.h"
#include "telescope/oskar_telescope.h"
#include "utility/oskar_getline.h"
#include "utility/oskar_string_to_array.h"
#include "math/oskar_cmath.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEG2RAD (M_PI / 180.0)

#ifdef __cplusplus
extern "C" {
#endif

static void set_coords(oskar_Station* station, int set_recursive,
        size_t current_depth, size_t num_sub_ids, int* sub_ids, int coord_type,
        double lon, double lat, int* status);

void oskar_telescope_load_pointing_file(oskar_Telescope* telescope,
        const char* filename, int* status)
{
    char* line = 0;
    size_t bufsize = 0, size_id = 0, i = 0, num_par = 0;
    FILE* file = 0;
    char** par = 0;
    int* id = 0;
    if (*status || !filename || strlen(filename) == 0) return;

    /* Check type. */
    const int type = oskar_telescope_precision(telescope);
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* Open the file. */
    file = fopen(filename, "r");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Loop over each line in the file. */
    while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
    {
        int coordsys = 0;
        char* end_ptr = 0;
        const int base = 10;

        /* Split into string array and check for required number of fields. */
        const size_t read = oskar_string_to_array_realloc_s(
                line, &num_par, &par);
        if (read < 4) continue;

        /* Get number of IDs. */
        const size_t num_ids = read - 3;

        /* Get all IDs on the line. */
        for (i = 0; i < num_ids && i < num_par; ++i)
        {
            /* Ensure enough space in ID array. */
            if (i >= size_id)
            {
                void* t = realloc(id, (size_id + 1) * sizeof(int));
                if (!t)
                {
                    *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
                    break;
                }
                id = (int*) t;
                ++size_id;
            }

            /* Store the ID, checking for '*' wildcard. */
            if (!par[i] || par[i][0] == '*')
            {
                id[i] = -1;
            }
            else
            {
                id[i] = (int) strtol(par[i], &end_ptr, base);
                if (end_ptr == par[i])
                {
                    *status = OSKAR_ERR_BAD_POINTING_FILE;
                    break;
                }
            }
        }
        if (*status) break;

        /* Get coordinate system type. */
        if (!par[i] || (par[i][0] != 'A' && par[i][0] != 'a'))
        {
            coordsys = OSKAR_COORDS_RADEC;
        }
        else
        {
            coordsys = OSKAR_COORDS_AZEL;
        }

        /* Get longitude and latitude values. */
        ++i;
        const double lon = strtod(par[i], &end_ptr) * DEG2RAD;
        if (end_ptr == par[i])
        {
            *status = OSKAR_ERR_BAD_POINTING_FILE;
            break;
        }
        ++i;
        const double lat = strtod(par[i], &end_ptr) * DEG2RAD;
        if (end_ptr == par[i])
        {
            *status = OSKAR_ERR_BAD_POINTING_FILE;
            break;
        }

        /* Set the data into the telescope model. */
        if ((num_ids > 0) && id)
        {
            int* sub_id = 0;
            if (num_ids > 1) sub_id = &id[1];
            if (id[0] < 0)
            {
                /* Loop over stations. */
                for (i = 0; i < (size_t)telescope->num_station_models; ++i)
                {
                    set_coords(oskar_telescope_station(telescope, (int)i), 0, 0,
                            num_ids - 1, sub_id, coordsys, lon, lat, status);
                }
            }
            else if (id[0] < telescope->num_station_models)
            {
                set_coords(oskar_telescope_station(telescope, id[0]), 0, 0,
                        num_ids - 1, sub_id, coordsys, lon, lat, status);
            }
            else
            {
                *status = OSKAR_ERR_BAD_POINTING_FILE;
                break;
            }
        }
        else
        {
            /* Could perhaps override interferometer phase centre here? */
            *status = OSKAR_ERR_BAD_POINTING_FILE;
            break;
        }
    }

    /* Free memory and close the file. */
    free(line);
    free(par);
    free(id);
    fclose(file);
}

static void set_coords(oskar_Station* station, int set_recursive,
        size_t current_depth, size_t num_sub_ids, int* sub_ids, int coord_type,
        double lon, double lat, int* status)
{
    if (*status || !station) return;

    if (set_recursive)
    {
        /* Set pointing data for this station. */
        oskar_station_set_phase_centre(station, coord_type, lon, lat);

        /* Set pointing data recursively for all child stations. */
        if (oskar_station_has_child(station))
        {
            size_t i = 0, num_elements = 0;
            num_elements = (size_t)oskar_station_num_elements(station);
            for (i = 0; i < num_elements; ++i)
            {
                set_coords(oskar_station_child(station, (int)i), 1,
                        current_depth + 1, num_sub_ids, sub_ids,
                        coord_type, lon, lat, status);
            }
        }
    }
    else
    {
        /* Check if there's nothing more in the list of indices. */
        if (current_depth == num_sub_ids)
        {
            set_coords(station, 1, current_depth,
                    num_sub_ids, sub_ids, coord_type, lon, lat, status);
        }
        else if (oskar_station_has_child(station))
        {
            int id = 0;

            /* Get the ID at this depth. */
            if (current_depth < num_sub_ids)
            {
                id = sub_ids[current_depth];
            }
            else
            {
                *status = OSKAR_ERR_BAD_POINTING_FILE;
                return;
            }

            if (id < 0)
            {
                size_t i = 0, num_elements = 0;
                num_elements = (size_t)oskar_station_num_elements(station);
                for (i = 0; i < num_elements; ++i)
                {
                    set_coords(oskar_station_child(station, (int)i), 0,
                            current_depth + 1, num_sub_ids, sub_ids,
                            coord_type, lon, lat, status);
                }
            }
            else if (id < oskar_station_num_elements(station))
            {
                set_coords(oskar_station_child(station, id), 0,
                        current_depth + 1, num_sub_ids, sub_ids,
                        coord_type, lon, lat, status);
            }
            else
            {
                *status = OSKAR_ERR_BAD_POINTING_FILE;
                return;
            }
        }
        else
        {
            *status = OSKAR_ERR_BAD_POINTING_FILE;
            return;
        }
    }
}

#ifdef __cplusplus
}
#endif
