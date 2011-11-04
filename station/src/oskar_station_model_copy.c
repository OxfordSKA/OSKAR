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

#include "station/oskar_station_model_copy.h"
#include "utility/oskar_mem_copy.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_station_model_copy(oskar_StationModel* dst,
        const oskar_StationModel* src)
{
    int error = 0;

    /* Check that all pointers are not NULL. */
    if (src == NULL || dst == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Copy the memory blocks. */
    error = oskar_mem_copy(&dst->x, &src->x);
    if (error) return error;
    error = oskar_mem_copy(&dst->y, &src->y);
    if (error) return error;
    error = oskar_mem_copy(&dst->z, &src->z);
    if (error) return error;
    error = oskar_mem_copy(&dst->weight, &src->weight);
    if (error) return error;
    error = oskar_mem_copy(&dst->amp_gain, &src->amp_gain);
    if (error) return error;
    error = oskar_mem_copy(&dst->amp_error, &src->amp_error);
    if (error) return error;
    error = oskar_mem_copy(&dst->phase_offset, &src->phase_offset);
    if (error) return error;
    error = oskar_mem_copy(&dst->phase_error, &src->phase_error);
    if (error) return error;

    /* Copy the meta data. */
    dst->n_elements = src->n_elements;
    dst->longitude = src->longitude;
    dst->latitude = src->latitude;
    dst->altitude = src->altitude;
    dst->ra0 = src->ra0;
    dst->dec0 = src->dec0;
    dst->single_element_model = src->single_element_model;
    dst->bit_depth = src->bit_depth;
    dst->coord_units = src->coord_units;

    /* TODO Work out how to deal with child stations. */
    /* TODO Work out how to deal with element pattern data. */

    return 0;
}

#ifdef __cplusplus
}
#endif
