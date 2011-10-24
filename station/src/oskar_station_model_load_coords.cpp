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
        double x, y, z, w_re, w_im, amp, amp_err, ph, ph_err;
        while (oskar_getline(&line, &n, file) != OSKAR_ERR_EOF)
        {
            // Ignore comment lines (lines starting with '#').
            if (line[0] == '#') continue;

            // Load element coordinates.
            int read = sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf %lf",
                    &x, &y, &z, &w_re, &w_im, &amp, &amp_err, &ph, &ph_err);
            if (read < 3) continue;

            // Ensure enough space in arrays.
            if (num_elements_loaded % 100 == 0)
                station->resize(num_elements_loaded + 100);

            // Store the data.
            ((double*)station->x.data)[num_elements_loaded] = x;
            ((double*)station->y.data)[num_elements_loaded] = y;
            ((double*)station->z.data)[num_elements_loaded] = z;
            ++num_elements_loaded;
        }
    }
    else if (station->x.type() == OSKAR_SINGLE)
    {
        float x, y, z, w_re, w_im, amp, amp_err, ph, ph_err;
        while (oskar_getline(&line, &n, file) != OSKAR_ERR_EOF)
        {
            // Ignore comment lines (lines starting with '#').
            if (line[0] == '#') continue;

            // Load element coordinates.
            int read = sscanf(line, "%f %f %f %f %f %f %f %f %f",
                    &x, &y, &z, &w_re, &w_im, &amp, &amp_err, &ph, &ph_err);
            if (read < 3) continue;

            // Ensure enough space in arrays.
            if (num_elements_loaded % 100 == 0)
                station->resize(num_elements_loaded + 100);

            // Store the data.
            ((float*)station->x.data)[num_elements_loaded] = x;
            ((float*)station->y.data)[num_elements_loaded] = y;
            ((float*)station->z.data)[num_elements_loaded] = z;
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
