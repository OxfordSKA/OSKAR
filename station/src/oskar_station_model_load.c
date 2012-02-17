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

#include "station/oskar_station_model_load.h"
#include "station/oskar_station_model_resize.h"
#include "station/oskar_station_model_set_element_errors.h"
#include "station/oskar_station_model_set_element_pos.h"
#include "station/oskar_station_model_set_element_weight.h"
#include "station/oskar_station_model_type.h"
#include "utility/oskar_getline.h"
#include "utility/oskar_string_to_array.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_station_model_load(oskar_StationModel* station, const char* filename)
{
    /* Declare the line buffer and counter. */
    char* line = NULL;
    size_t bufsize = 0;
    int n = 0, type = 0, err = 0;
    FILE* file;

    /* Check that all pointers are not NULL. */
    if (station == NULL || filename == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Check type. */
    type = oskar_station_model_type(station);
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
        return OSKAR_ERR_BAD_DATA_TYPE;

    /* Open the file. */
    file = fopen(filename, "r");
    if (!file)
        return OSKAR_ERR_FILE_IO;

    /* Loop over each line in the file. */
    while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
    {
        /* Declare parameter array. */
        double par[] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
        int read = 0;

        /* Ignore comment lines (lines starting with '#'). */
        if (line[0] == '#') continue;

        /* Load element coordinates. */
        read = oskar_string_to_array_d(line, 9, par);
        if (read < 2) continue;

        /* Ensure enough space in arrays. */
        if (n % 100 == 0)
        {
            err = oskar_station_model_resize(station, n + 100);
            if (err)
            {
                fclose(file);
                return err;
            }
        }

        /* Store the data. */
        err = oskar_station_model_set_element_pos(station, n,
                par[0], par[1], par[2]);
        if (err)
        {
            fclose(file);
            return err;
        }
        err = oskar_station_model_set_element_errors(station, n,
                par[3], par[4], par[5], par[6]);
        if (err)
        {
            fclose(file);
            return err;
        }
        err = oskar_station_model_set_element_weight(station, n,
                par[7], par[8]);
        if (err)
        {
            fclose(file);
            return err;
        }

        ++n;
    }

    /* Record the number of elements loaded. */
    station->num_elements = n;

    /* Set the coordinate units to metres. */
    station->coord_units = OSKAR_METRES;

    /* Free the line buffer and close the file. */
    if (line) free(line);
    fclose(file);

    return 0;
}

#ifdef __cplusplus
}
#endif
