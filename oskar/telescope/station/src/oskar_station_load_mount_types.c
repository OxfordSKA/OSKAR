/*
 * Copyright (c) 2015, The University of Oxford
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

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_load_mount_types(oskar_Station* station,
        const char* filename, int* status)
{
    /* Declare the line buffer and counter. */
    char* line = NULL;
    size_t bufsize = 0;
    int n = 0, type = 0, old_size = 0;
    FILE* file;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check type. */
    type = oskar_station_precision(station);
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

    /* Get the size of the station before loading the data. */
    old_size = oskar_station_num_elements(station);

    /* Loop over each line in the file. */
    while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
    {
        char type = '\0';
        if (bufsize > 0) type = line[0];
        type = toupper(type);
        if (type != 'F' && type != 'A')
            continue;

        /* Ensure the station model is big enough. */
        if (oskar_station_num_elements(station) <= n)
        {
            oskar_station_resize(station, n + 100, status);
            if (*status) break;
        }

        /* Store the data. */
        oskar_station_set_element_mount_type(station, n, type, status);

        /* Increment element counter. */
        ++n;
    }

    /* Consistency check with previous station size (should be the same as
     * the number of elements loaded). */
    if (!*status && n != old_size)
        *status = OSKAR_ERR_DIMENSION_MISMATCH;

    /* Free the line buffer and close the file. */
    free(line);
    fclose(file);
}

#ifdef __cplusplus
}
#endif
