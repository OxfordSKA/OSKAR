/*
 * Copyright (c) 2013-2015, The University of Oxford
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
    /* Declare the line buffer and counter. */
    char* line = NULL;
    size_t bufsize = 0, size_id = 0, num_par = 0;
    int type = 0;
    FILE* file;
    char** par = 0;
    int* id = 0;

    /* Check if safe to proceed. */
    if (*status || !filename || strlen(filename) == 0) return;

    /* Check type. */
    type = oskar_telescope_precision(telescope);
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
        int coordsys;
        size_t i = 0, num_ids = 0, read = 0;
        double lon = 0.0, lat = 0.0;

        /* Split into string array and check for required number of fields. */
        read = oskar_string_to_array_realloc_s(line, &num_par, &par);
        if (read < 4) continue;

        /* Get number of IDs. */
        num_ids = read - 3;

        /* Get all IDs on the line. */
        for (i = 0; i < num_ids && i < num_par; ++i)
        {
            /* Ensure enough space in ID array. */
            if (i >= size_id)
            {
                void* t;
                t = realloc(id, (size_id + 1) * sizeof(int));
                if (!t)
                {
                    *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
                    break;
                }
                id = t;
                ++size_id;
            }

            /* Store the ID, checking for '*' wildcard. */
            if (!par[i] || par[i][0] == '*')
                id[i] = -1;
            else
                sscanf(par[i], "%d", &(id[i]));
        }
        if (*status) break;

        /* Get coordinate system type. */
        if (!par[i] || (par[i][0] != 'A' && par[i][0] != 'a'))
            coordsys = OSKAR_SPHERICAL_TYPE_EQUATORIAL;
        else
            coordsys = OSKAR_SPHERICAL_TYPE_AZEL;

        /* Get longitude and latitude values. */
        ++i;
        sscanf(par[i], "%lf", &lon);
        ++i;
        sscanf(par[i], "%lf", &lat);

        /* Set the data into the telescope model. */
        lon *= DEG2RAD;
        lat *= DEG2RAD;
        if (num_ids > 0)
        {
            int* sub_id = 0;
            if (num_ids > 1) sub_id = &id[1];
            if (id[0] < 0)
            {
                /* Loop over stations. */
                for (i = 0; i < (size_t)telescope->num_stations; ++i)
                {
                    set_coords(oskar_telescope_station(telescope, (int)i), 0, 0,
                            num_ids - 1, sub_id, coordsys, lon, lat, status);
                }
            }
            else if (id[0] < telescope->num_stations)
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

    /* Free the line buffer and close the file. */
    if (line) free(line);
    if (par) free(par);
    if (id) free(id);
    fclose(file);
}

static void set_coords(oskar_Station* station, int set_recursive,
        size_t current_depth, size_t num_sub_ids, int* sub_ids, int coord_type,
        double lon, double lat, int* status)
{
    /* Check if safe to proceed. */
    if (*status) return;

    if (set_recursive)
    {
        /* Set pointing data for this station. */
        oskar_station_set_phase_centre(station, coord_type, lon, lat);

        /* Set pointing data recursively for all child stations. */
        if (oskar_station_has_child(station))
        {
            size_t i, num_elements;
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
            int id;

            /* Get the ID at this depth. */
            if (current_depth < num_sub_ids)
                id = sub_ids[current_depth];
            else
            {
                *status = OSKAR_ERR_BAD_POINTING_FILE;
                return;
            }

            if (id < 0)
            {
                size_t i, num_elements;
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
