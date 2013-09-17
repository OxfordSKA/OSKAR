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

#include "station/oskar_work_station_beam_init.h"
#include <oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_work_station_beam_init(oskar_WorkStationBeam* work, int type,
        int location, int* status)
{
    int i, complex_matrix;

    /* Check all inputs. */
    if (!work || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check the base type is correct. */
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
        *status = OSKAR_ERR_BAD_DATA_TYPE;

    /* Set complex matrix type. */
    complex_matrix = OSKAR_COMPLEX | OSKAR_MATRIX;

    /* Initialise arrays. */
    oskar_mem_init(&work->horizon_mask, OSKAR_INT, location, 0, 1, status);
    oskar_mem_init(&work->theta_modified, type, location, 0, 1, status);
    oskar_mem_init(&work->phi_modified, type, location, 0, 1, status);
    oskar_mem_init(&work->hor_x, type, location, 0, 1, status);
    oskar_mem_init(&work->hor_y, type, location, 0, 1, status);
    oskar_mem_init(&work->hor_z, type, location, 0, 1, status);
    oskar_mem_init(&work->weights, (type | OSKAR_COMPLEX),
            location, 0, 1, status);
    oskar_mem_init(&work->weights_error, (type | OSKAR_COMPLEX),
            location, 0, 1, status);
    oskar_mem_init(&work->array_pattern, (type | OSKAR_COMPLEX),
            location, 0, 1, status);
    oskar_mem_init(&work->element_pattern_matrix, (type | complex_matrix),
            location, 0, 1, status);
    oskar_mem_init(&work->element_pattern_scalar, (type | OSKAR_COMPLEX),
            location, 0, 1, status);

    for (i = 0; i < OSKAR_MAX_STATION_DEPTH; ++i)
    {
        oskar_mem_init(&work->hierarchy_work_matrix[i], (type | complex_matrix),
                location, 0, 1, status);
        oskar_mem_init(&work->hierarchy_work_scalar[i], (type | OSKAR_COMPLEX),
                location, 0, 1, status);
    }
}

#ifdef __cplusplus
}
#endif
