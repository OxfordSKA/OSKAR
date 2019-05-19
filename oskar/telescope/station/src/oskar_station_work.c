/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include "telescope/station/oskar_station_work.h"
#include "telescope/station/private_station_work.h"

#ifdef __cplusplus
extern "C" {
#endif

static void get_mem_from_template(oskar_Mem** b, const oskar_Mem* a,
        size_t length, int* status);

oskar_StationWork* oskar_station_work_create(int type,
        int location, int* status)
{
    oskar_StationWork* work = 0;

    /* Allocate memory for the structure. */
    work = (oskar_StationWork*) calloc(1, sizeof(oskar_StationWork));

    /* Check the base type is correct. */
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
        *status = OSKAR_ERR_BAD_DATA_TYPE;

    /* Initialise arrays. */
    work->horizon_mask = oskar_mem_create(OSKAR_INT, location, 0, status);
    work->source_indices = oskar_mem_create(OSKAR_INT, location, 0, status);
    work->theta_modified = oskar_mem_create(type, location, 0, status);
    work->phi_modified = oskar_mem_create(type, location, 0, status);
    work->enu_direction_x = oskar_mem_create(type, location, 0, status);
    work->enu_direction_y = oskar_mem_create(type, location, 0, status);
    work->enu_direction_z = oskar_mem_create(type, location, 0, status);
    work->weights = oskar_mem_create((type | OSKAR_COMPLEX),
            location, 0, status);
    work->weights_error = oskar_mem_create((type | OSKAR_COMPLEX),
            location, 0, status);
    work->array_pattern = oskar_mem_create((type | OSKAR_COMPLEX),
            location, 0, status);
    work->beam_out_scratch = 0;
    work->num_depths = 0;
    work->beam = 0;

    return work;
}

void oskar_station_work_free(oskar_StationWork* work, int* status)
{
    int i;
    if (!work) return;

    oskar_mem_free(work->horizon_mask, status);
    oskar_mem_free(work->source_indices, status);
    oskar_mem_free(work->theta_modified, status);
    oskar_mem_free(work->phi_modified, status);
    oskar_mem_free(work->enu_direction_x, status);
    oskar_mem_free(work->enu_direction_y, status);
    oskar_mem_free(work->enu_direction_z, status);
    oskar_mem_free(work->weights, status);
    oskar_mem_free(work->weights_error, status);
    oskar_mem_free(work->array_pattern, status);
    oskar_mem_free(work->beam_out_scratch, status);

    for (i = 0; i < work->num_depths; ++i)
    {
        oskar_mem_free(work->beam[i], status);
    }

    /* Free the structure. */
    free(work);
}

oskar_Mem* oskar_station_work_horizon_mask(oskar_StationWork* work)
{
    return work->horizon_mask;
}

oskar_Mem* oskar_station_work_source_indices(oskar_StationWork* work)
{
    return work->source_indices;
}

oskar_Mem* oskar_station_work_enu_direction_x(oskar_StationWork* work)
{
    return work->enu_direction_x;
}

oskar_Mem* oskar_station_work_enu_direction_y(oskar_StationWork* work)
{
    return work->enu_direction_y;
}

oskar_Mem* oskar_station_work_enu_direction_z(oskar_StationWork* work)
{
    return work->enu_direction_z;
}

oskar_Mem* oskar_station_work_beam_out(oskar_StationWork* work,
        const oskar_Mem* output_beam, size_t length, int* status)
{
    get_mem_from_template(&work->beam_out_scratch, output_beam,
            1 + length, status);
    return work->beam_out_scratch;
}

oskar_Mem* oskar_station_work_beam(oskar_StationWork* work,
        const oskar_Mem* output_beam, size_t length, int depth, int* status)
{
    if (depth > work->num_depths - 1)
    {
        int i, old_num_depths;
        old_num_depths = work->num_depths;
        work->num_depths = depth + 1;
        work->beam = (oskar_Mem**) realloc(work->beam,
                work->num_depths * sizeof(oskar_Mem*));
        for (i = old_num_depths; i < work->num_depths; ++i)
        {
            work->beam[i] = 0;
        }
    }

    get_mem_from_template(&work->beam[depth], output_beam, length, status);
    return work->beam[depth];
}

static void get_mem_from_template(oskar_Mem** b, const oskar_Mem* a,
        size_t length, int* status)
{
    const int type = oskar_mem_type(a);
    const int loc = oskar_mem_location(a);

    /* Check if the array exists with an incorrect type and location. */
    if (*b && (oskar_mem_type(*b) != type || oskar_mem_location(*b) != loc))
    {
        oskar_mem_free(*b, status);
        *b = 0;
    }

    /* Create or resize the array. */
    if (!*b)
        *b = oskar_mem_create(type, loc, length, status);
    else
        oskar_mem_ensure(*b, length, status);
}

#ifdef __cplusplus
}
#endif
