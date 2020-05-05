/*
 * Copyright (c) 2019-2020, The University of Oxford
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

#include "telescope/station/oskar_station.h"

#include "utility/oskar_getline.h"
#include "utility/oskar_string_to_array.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_load_cable_length_error(oskar_Station* station, int feed,
        const char* filename, int* status)
{
    char* line = NULL;
    size_t bufsize = 0;
    int n = 0;
    FILE* file;
    if (*status || !station) return;
    const int type = oskar_station_precision(station);
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    file = fopen(filename, "r");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }
    const int old_size = oskar_station_num_elements(station);
    while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
    {
        double par[] = {0.0};
        size_t num_par = sizeof(par) / sizeof(double);
        if (oskar_string_to_array_d(line, num_par, par) < 1) continue;
        if (oskar_station_num_elements(station) <= n)
        {
            oskar_station_resize(station, n + 256, status);
            if (*status) break;
        }
        oskar_station_set_element_cable_length_error(station, feed,
                n, par[0], status);
        ++n;
    }

    /* Consistency check with previous station size (should be the same as
     * the number of elements loaded). */
    if (!*status && n != old_size)
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
    free(line);
    fclose(file);
}

#ifdef __cplusplus
}
#endif
