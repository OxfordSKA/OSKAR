/*
 * Copyright (c) 2013, The University of Oxford
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


#include <oskar_WorkJonesZ.h>
#include <oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_work_jones_z_init(oskar_WorkJonesZ* work, int type, int location,
        int* status)
{
    /* Check inputs */
    if (!work || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check the base type is correct */
    if (!(type == OSKAR_SINGLE || type == OSKAR_DOUBLE))
        *status = OSKAR_ERR_BAD_DATA_TYPE;

    oskar_mem_init(&work->hor_x, type, location, 0, 1, status);
    oskar_mem_init(&work->hor_y, type, location, 0, 1, status);
    oskar_mem_init(&work->hor_z, type, location, 0, 1, status);
    oskar_mem_init(&work->pp_lon, type, location, 0, 1, status);
    oskar_mem_init(&work->pp_lat, type, location, 0, 1, status);
    oskar_mem_init(&work->pp_rel_path, type, location, 0, 1, status);
    oskar_mem_init(&work->screen_TEC, type, location, 0, 1, status);
    oskar_mem_init(&work->total_TEC, type, location, 0, 1, status);
}


void oskar_work_jones_z_free(oskar_WorkJonesZ* work, int* status)
{
    if (!work || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    oskar_mem_free(&work->hor_x, status);
    oskar_mem_free(&work->hor_y, status);
    oskar_mem_free(&work->hor_z, status);
    oskar_mem_free(&work->pp_lon, status);
    oskar_mem_free(&work->pp_lat, status);
    oskar_mem_free(&work->pp_rel_path, status);
    oskar_mem_free(&work->screen_TEC, status);
    oskar_mem_free(&work->total_TEC, status);
}

int oskar_work_jones_z_type(oskar_WorkJonesZ* work)
{
    if (oskar_mem_type(&work->hor_x) == OSKAR_DOUBLE &&
            oskar_mem_type(&work->hor_y) == OSKAR_DOUBLE &&
            oskar_mem_type(&work->hor_z) == OSKAR_DOUBLE &&
            oskar_mem_type(&work->pp_lon) == OSKAR_DOUBLE &&
            oskar_mem_type(&work->pp_lat) == OSKAR_DOUBLE &&
            oskar_mem_type(&work->pp_rel_path) == OSKAR_DOUBLE &&
            oskar_mem_type(&work->screen_TEC) == OSKAR_DOUBLE &&
            oskar_mem_type(&work->total_TEC) == OSKAR_DOUBLE)
    {
        return OSKAR_DOUBLE;
    }
    else if (oskar_mem_type(&work->hor_x) == OSKAR_SINGLE &&
            oskar_mem_type(&work->hor_y) == OSKAR_SINGLE &&
            oskar_mem_type(&work->hor_z) == OSKAR_SINGLE &&
            oskar_mem_type(&work->pp_lon) == OSKAR_SINGLE &&
            oskar_mem_type(&work->pp_lat) == OSKAR_SINGLE &&
            oskar_mem_type(&work->pp_rel_path) == OSKAR_SINGLE &&
            oskar_mem_type(&work->screen_TEC) == OSKAR_SINGLE &&
            oskar_mem_type(&work->total_TEC) == OSKAR_SINGLE)
    {
        return OSKAR_SINGLE;
    }
    return -1;
}

void oskar_work_jones_z_resize(oskar_WorkJonesZ* work, int n, int* status)
{
    if (!work || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }
    oskar_mem_realloc(&work->hor_x, n, status);
    oskar_mem_realloc(&work->hor_y, n, status);
    oskar_mem_realloc(&work->hor_z, n, status);
    oskar_mem_realloc(&work->pp_lon, n, status);
    oskar_mem_realloc(&work->pp_lat, n, status);
    oskar_mem_realloc(&work->pp_rel_path, n, status);
    oskar_mem_realloc(&work->screen_TEC, n, status);
    oskar_mem_realloc(&work->total_TEC, n, status);
}


#ifdef __cplusplus
}
#endif
