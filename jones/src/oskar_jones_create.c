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

#include <private_jones.h>
#include <oskar_jones.h>
#include <oskar_mem.h>

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Jones* oskar_jones_create(int type, int location, int num_stations,
        int num_sources, int* status)
{
    oskar_Jones* jones = 0;
    int base_type;
    size_t n_elements;

    /* Check type and location. */
    base_type = oskar_type_precision(type);
    if (base_type != OSKAR_SINGLE && base_type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return 0;
    }
    if (!oskar_type_is_complex(type))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return 0;
    }
    if (location != OSKAR_CPU && location != OSKAR_GPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return 0;
    }

    /* Allocate and initialise the structure. */
    jones = (oskar_Jones*) malloc(sizeof(oskar_Jones));
    if (!jones)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        return 0;
    }
    n_elements = num_stations * num_sources;
    jones->num_stations = num_stations;
    jones->num_sources = num_sources;
    jones->cap_stations = num_stations;
    jones->cap_sources = num_sources;
    jones->data = oskar_mem_create(type, location, n_elements, status);

    /* Return pointer to the structure. */
    return jones;
}

#ifdef __cplusplus
}
#endif
