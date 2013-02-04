/*
 * Copyright (c) 2013, The University of Oxford
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

#include "station/oskar_station_model_load_pointing_file.h"
#include "station/oskar_station_model_type.h"
#include "utility/oskar_getline.h"
#include "utility/oskar_string_to_array.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI  3.14159265358979323846264338327950288
#endif

#define DEG2RAD (M_PI / 180.0)

#ifdef __cplusplus
extern "C" {
#endif

static void set_coords(oskar_StationModel* station, int current_depth,
        int depth, int id, double lon, double lat, int coord_type,
        int* status);

void oskar_station_model_load_pointing_file(oskar_StationModel* station,
        const char* filename, int* status)
{
    /* Declare the line buffer and counter. */
    char* line = NULL;
    size_t bufsize = 0;
    int type = 0;
    FILE* file;

    /* Check all inputs. */
    if (!station || !filename || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check type. */
    type = oskar_station_model_type(station);
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
        char *par[] = {0, 0, 0, 0, 0};
        int depth = 0, id = 0, coordsys = 0, num_par = 0;
        double lon = 0.0, lat = 0.0;

        /* Split into string array and check for required number of fields. */
        num_par = sizeof(par) / sizeof(char*);
        if (oskar_string_to_array_s(line, num_par, par) < num_par) continue;

        /* Get depth and station IDs, and check for '*' wildcards. */
        if (!par[0] || par[0][0] == '*')
            depth = -1;
        else
            sscanf(par[0], "%d", &depth);
        if (!par[1] || par[1][0] == '*')
            id = -1;
        else
            sscanf(par[1], "%d", &id);

        /* Get longitude and latitude values. */
        sscanf(par[2], "%lf", &lon);
        sscanf(par[3], "%lf", &lat);

        /* Get coordinate system type. */
        if (!par[4] || (par[4][0] != 'A' && par[4][0] != 'a'))
            coordsys = OSKAR_SPHERICAL_TYPE_EQUATORIAL;
        else
            coordsys = OSKAR_SPHERICAL_TYPE_HORIZONTAL;

        /* Set the data into the station model. */
        set_coords(station, 0, depth, id, lon * DEG2RAD, lat * DEG2RAD,
                coordsys, status);
    }

    /* Free the line buffer and close the file. */
    if (line) free(line);
    fclose(file);
}

static void set_coords(oskar_StationModel* station, int current_depth,
        int depth, int id, double lon, double lat, int coord_type, int* status)
{
    /* Check if safe to proceed. */
    if (*status) return;

    if (depth < 0 || current_depth == depth)
    {
        /* Set pointing data for this station. */
        station->beam_coord_type = coord_type;
        station->beam_longitude_rad = lon;
        station->beam_latitude_rad = lat;
    }

    /* Check if we need to go deeper. */
    if (station->child && (depth < 0 || current_depth < depth))
    {
        /* Descend deeper. */
        int i, start, end;
        start = id;
        end = id;

        if (start < 0 || end < 0)
        {
            start = 0;
            end = station->num_elements - 1;
        }

        for (i = start; i <= end; ++i)
        {
            /* Range check! */
            if (i < station->num_elements)
            {
                set_coords(&station->child[i], current_depth + 1,
                        depth, id, lon, lat, coord_type, status);
            }
        }
    }
}


#ifdef __cplusplus
}
#endif
