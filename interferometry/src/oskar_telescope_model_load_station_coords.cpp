/*
 * Copyright (c) 2011, The University of Oxford
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

#include "interferometry/oskar_telescope_model_load_station_coords.h"
#include "interferometry/oskar_TelescopeModel.h"
#include "station/oskar_StationModel.h"
#include "utility/oskar_getline.h"
#include "utility/oskar_string_to_array.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>

int oskar_telescope_model_load_station_coords(oskar_TelescopeModel* telescope,
        const char* filename, const double longitude, const double latitude)
{
    // Check that all pointers are not NULL.
    if (telescope == NULL || filename == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    // Check that the data is in the right location.
    // FIXME some way of effectively locking the location for the entire structure?
    // TODO check location ?

    // Open the coordinate file.
    FILE* file = fopen(filename, "r");
    if (!file)
        return OSKAR_ERR_FILE_IO;

    // Declare the line buffer.
    char* line = NULL;
    size_t bufsize = 0;

    // Loop over each line in the file to read out coordinates.
    int n = 0;
    if (telescope->station_x.type() == OSKAR_DOUBLE)
    {
        // Declare temporary arrays for loading horizontal coordinates.
        double* hor_x = NULL;
        double* hor_y = NULL;
        double* hor_z = NULL;

        while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
        {
            // Ignore comment lines (lines starting with '#').
            if (line[0] == '#') continue;

            // Load coordinates.
            double par[] = {0.0, 0.0, 0.0}; // x, y, z
            int read = oskar_string_to_array_d(line, 3, par);
            if (read < 2) continue;

            // Ensure enough space in arrays.
            if (n % 100 == 0)
            {
                size_t mem_size = (n + 100) * sizeof(double);
                hor_x = (double*)realloc((void*)hor_x, mem_size);
                hor_y = (double*)realloc((void*)hor_y, mem_size);
                hor_z = (double*)realloc((void*)hor_z, mem_size);
            }

            // Store the data.
            hor_x[n] = par[0];
            hor_y[n] = par[1];
            hor_z[n] = par[2];
            ++n;
        }

        // Record the number of station positions loaded.
        telescope->num_stations = n;

        // Allocate memory for ITRS station coordinates.
        telescope->station_x.resize(n);
        telescope->station_y.resize(n);
        telescope->station_z.resize(n);

        // TODO convert horizontal coordinates + long, lat to ITRS (station_x,
        // station_y, station_z).

        // Free memory used to store horizontal coordinates.
        free(hor_x);
        free(hor_y);
        free(hor_z);
    }
    else if (telescope->station_x.type() == OSKAR_SINGLE)
    {
        // Declare temporary arrays for loading horizontal coordinates.
        float* hor_x = NULL;
        float* hor_y = NULL;
        float* hor_z = NULL;

        while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
        {
            // Ignore comment lines (lines starting with '#').
            if (line[0] == '#') continue;

            // Load coordinates.
            float par[] = {0.0, 0.0, 0.0}; // x, y, z
            int read = oskar_string_to_array_f(line, 3, par);
            if (read < 2) continue;

            // Ensure enough space in arrays.
            if (n % 100 == 0)
            {
                size_t mem_size = (n + 100) * sizeof(float);
                hor_x = (float*) realloc(hor_x, mem_size);
                hor_y = (float*) realloc(hor_y, mem_size);
                hor_z = (float*) realloc(hor_z, mem_size);
            }

            // Store the data.
            hor_x[n] = par[0];
            hor_y[n] = par[1];
            hor_z[n] = par[2];
            ++n;
        }

        // Record the number of station positions loaded.
        telescope->num_stations = n;

        // Allocate memory for ITRS station coordinates.
        telescope->station_x.resize(n);
        telescope->station_y.resize(n);
        telescope->station_z.resize(n);

        // TODO convert horizontal coordinates + long, lat to ITRS (station_x,
        // station_y, station_z).

        // Free memory used to store horizontal coordinates.
        free(hor_x);
        free(hor_y);
        free(hor_z);
    }
    else
    {
        fclose(file);
        return OSKAR_ERR_BAD_DATA_TYPE;
    }

    // Allocate the station structure array
    // TODO probably better to use a constructor on StationModel here?
//    telescope->station = new oskar_StationModel[n];
    telescope->station = (oskar_StationModel*)malloc(n * sizeof(oskar_StationModel));

    // Free the line buffer and close the file.
    if (line) free(line);
    fclose(file);

    return 0;
}

