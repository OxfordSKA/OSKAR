/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#include <private_station.h>
#include <oskar_station.h>

#include <oskar_getline.h>
#include <oskar_string_to_array.h>

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_load_config(oskar_Station* station,
        const char* filename, int* status)
{
    /* Declare the line buffer and counter. */
    char* line = NULL;
    size_t bufsize = 0;
    int n = 0, type = 0;
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
    type = oskar_station_type(station);
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
        /* Declare parameter array. */
        /* x, y, z, delta_x, delta_y, delta_z, amp_gain, amp_error,
         * phase_offset_deg, phase_error_deg, weight_re, weight_im,
         * x_azimuth_deg, y_azimuth_deg */
        double par[] = {0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.,
                90., 0.};
        int num_par = sizeof(par) / sizeof(double);

        /* Load element coordinates. */
        if (oskar_string_to_array_d(line, num_par, par) < 2) continue;

        /* Ensure enough space in arrays. */
        if (n % 100 == 0)
            oskar_station_resize(station, n + 100, status);

        /* Store the data. */
        oskar_station_set_element_coords(station, n,
                par[0], par[1], par[2], par[3], par[4], par[5], status);
        oskar_station_set_element_errors(station, n,
                par[6], par[7], par[8], par[9], status);
        oskar_station_set_element_weight(station, n,
                par[10], par[11], status);
        oskar_station_set_element_orientation(station, n,
                par[12], par[13], status);

        /* Increment element counter. */
        ++n;
    }

    /* Record number of elements loaded, and set coordinate units to metres. */
    if (!(*status))
    {
        station->num_elements = n;
        station->coord_units = OSKAR_METRES;
    }

    /* Free the line buffer and close the file. */
    if (line) free(line);
    fclose(file);
}

#ifdef __cplusplus
}
#endif
