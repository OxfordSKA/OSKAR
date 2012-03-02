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


#include "station/oskar_station_model_write_coords.h"
#include "stdio.h"
#include "stdlib.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_station_model_write_coords(const char* filename,
        const oskar_StationModel* station)
{
    int i;
    FILE* file;

    if (filename == NULL || station == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    if (station->x.location == OSKAR_LOCATION_GPU ||
            station->y.location == OSKAR_LOCATION_GPU ||
            station->z.location == OSKAR_LOCATION_GPU)
    {
        return OSKAR_ERR_BAD_LOCATION;
    }

    if (station->coord_units != OSKAR_METRES)
        return OSKAR_ERR_BAD_UNITS;

    file = fopen(filename, "w");
    if (!file)
        return OSKAR_ERR_FILE_IO;

    fprintf(file, "# num_antennas = %i\n", station->num_elements);
    fprintf(file, "# longitude (radians) = %f\n", station->longitude_rad);
    fprintf(file, "# latitude (radians)  = %f\n", station->latitude_rad);
    fprintf(file, "# altitude (metres)   = %f\n", station->altitude_metres);
    fprintf(file, "# local horizontal x(east), y(north), z(zenith) [metres]\n");
    if (station->x.type == OSKAR_DOUBLE &&
            station->y.type == OSKAR_DOUBLE &&
            station->z.type == OSKAR_DOUBLE)
    {
        for (i = 0; i < station->num_elements; ++i)
        {
            fprintf(file, "% -12.6e,% -12.6e,% -12.6e\n",
                    ((double*)station->x.data)[i],
                    ((double*)station->y.data)[i],
                    ((double*)station->z.data)[i]);
        }
    }
    else if (station->x.type == OSKAR_SINGLE &&
            station->y.type == OSKAR_SINGLE &&
            station->z.type == OSKAR_SINGLE)
    {
        for (i = 0; i < station->num_elements; ++i)
        {
            fprintf(file, "% -12.6e,% -12.6e,% -12.6e\n",
                    ((float*)station->x.data)[i],
                    ((float*)station->y.data)[i],
                    ((float*)station->z.data)[i]);
        }
    }
    else
    {
        return OSKAR_ERR_BAD_DATA_TYPE;
    }

    fclose(file);
    return 0;
}

#ifdef __cplusplus
}
#endif
