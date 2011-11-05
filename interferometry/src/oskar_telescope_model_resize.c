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

#include "interferometry/oskar_telescope_model_resize.h"
#include "interferometry/oskar_TelescopeModel.h"
#include "station/oskar_station_model_init.h"
#include "utility/oskar_mem_realloc.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_telescope_model_resize(oskar_TelescopeModel* telescope, int n_stations)
{
    int error = 0, old_size = 0;

    /* Sanity check on inputs. */
    if (telescope == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Get the old size. */
    old_size = telescope->num_stations;

    /* Resize the station array. */
    telescope->station = realloc(telescope->station,
            n_stations * sizeof(oskar_StationModel));
    if (n_stations > old_size)
    {
        /* Initialise new stations. */
        int i = 0;
        for (i = old_size; i < n_stations; ++i)
        {
            error = oskar_station_model_init(&(telescope->station[i]),
                    telescope->station_x.private_type,
                    telescope->station_x.private_location, 0);
            if (error) return error;
        }
    }

    /* Resize the remaining arrays. */
    error = oskar_mem_realloc(&(telescope->station_x), n_stations);
    if (error) return error;
    error = oskar_mem_realloc(&(telescope->station_y), n_stations);
    if (error) return error;
    error = oskar_mem_realloc(&(telescope->station_z), n_stations);
    if (error) return error;

    /* Store the new size. */
    telescope->num_stations = n_stations;

    return error;
}

#ifdef __cplusplus
}
#endif
