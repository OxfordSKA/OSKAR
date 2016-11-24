/*
 * Copyright (c) 2013-2015, The University of Oxford
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

#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_create_child_stations(oskar_Station* station,
        int* status)
{
    int i, type, location;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check that the memory isn't already allocated. */
    if (station->child)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        return;
    }

    /* Allocate memory for child station array. */
    station->child = malloc(station->num_elements * sizeof(oskar_Station*));
    if (!station->child)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        return;
    }

    /* Create and initialise each child station. */
    type = oskar_station_precision(station);
    location = oskar_station_mem_location(station);
    for (i = 0; i < station->num_elements; ++i)
    {
        station->child[i] = oskar_station_create(type, location, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
