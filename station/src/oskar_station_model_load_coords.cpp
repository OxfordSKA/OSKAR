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

#include "station/oskar_station_model_load_coords.h"
#include "utility/oskar_getline.h"
#include "utility/oskar_string_to_array.h"
#include <cstdio>

extern "C"
int oskar_station_model_load_coords(const char* filename,
        oskar_StationModel* station)
{
    // Check that all pointers are not NULL.
    if (station == NULL || filename == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    // Check that the data is in the right location.
    if (station->x.location() != OSKAR_LOCATION_CPU ||
            station->y.location() != OSKAR_LOCATION_CPU ||
            station->z.location() != OSKAR_LOCATION_CPU)
        return OSKAR_ERR_BAD_LOCATION;

    // Open the file.
    FILE* file = fopen(filename, "r");
    if (!file)
        return OSKAR_ERR_FILE_IO;

    // Declare the line buffer.
    char* line = NULL;
    size_t n = 0;

    // Loop over each line in the file.
    int num_elements_loaded = 0;
    if (station->x.type() == OSKAR_DOUBLE)
    {
        while (oskar_getline(&line, &n, file) != OSKAR_ERR_EOF)
        {
            // Ignore comment lines (lines starting with '#').
            if (line[0] == '#') continue;

            // Load element coordinates.
            double p[] = {0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0};
            int read = oskar_string_to_array_d(line, 9, p);
            if (read < 3) continue;

            // Ensure enough space in arrays.
            if (num_elements_loaded % 100 == 0)
                station->resize(num_elements_loaded + 100);

            // Store the data.
            ((double*)station->x.data)[num_elements_loaded] = p[0];
            ((double*)station->y.data)[num_elements_loaded] = p[1];
            ((double*)station->z.data)[num_elements_loaded] = p[2];
            ((double2*)station->weight.data)[num_elements_loaded].x = p[3];
            ((double2*)station->weight.data)[num_elements_loaded].y = p[4];
            ((double*)station->amp_gain.data)[num_elements_loaded] = p[5];
            ((double*)station->amp_error.data)[num_elements_loaded] = p[6];
            ((double*)station->phase_offset.data)[num_elements_loaded] = p[7];
            ((double*)station->phase_error.data)[num_elements_loaded] = p[8];
            ++num_elements_loaded;
        }
    }
    else if (station->x.type() == OSKAR_SINGLE)
    {
        while (oskar_getline(&line, &n, file) != OSKAR_ERR_EOF)
        {
            // Ignore comment lines (lines starting with '#').
            if (line[0] == '#') continue;

            // Load element coordinates.
            float p[] = {0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0};
            int read = oskar_string_to_array_f(line, 9, p);
            if (read < 3) continue;

            // Ensure enough space in arrays.
            if (num_elements_loaded % 100 == 0)
                station->resize(num_elements_loaded + 100);

            // Store the data.
            ((float*)station->x.data)[num_elements_loaded] = p[0];
            ((float*)station->y.data)[num_elements_loaded] = p[1];
            ((float*)station->z.data)[num_elements_loaded] = p[2];
            ((float2*)station->weight.data)[num_elements_loaded].x = p[3];
            ((float2*)station->weight.data)[num_elements_loaded].y = p[4];
            ((float*)station->amp_gain.data)[num_elements_loaded] = p[5];
            ((float*)station->amp_error.data)[num_elements_loaded] = p[6];
            ((float*)station->phase_offset.data)[num_elements_loaded] = p[7];
            ((float*)station->phase_error.data)[num_elements_loaded] = p[8];
            ++num_elements_loaded;
        }
    }
    else
    {
        fclose(file);
        return OSKAR_ERR_BAD_DATA_TYPE;
    }

    station->n_elements = num_elements_loaded;

    // Free the line buffer and close the file.
    if (line) free(line);
    fclose(file);

    return 0;
}
