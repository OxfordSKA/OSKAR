/*
 * Copyright (c) 2012, The University of Oxford
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

#include "interferometry/oskar_evaluate_baselines.h"
#include "interferometry/oskar_compute_baselines.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_evaluate_baselines(oskar_Mem* baseline_u, oskar_Mem* baseline_v,
        oskar_Mem* baseline_w, const oskar_Mem* station_u,
        const oskar_Mem* station_v, const oskar_Mem* station_w)
{
    int type, num_stations, num_baselines, error = 0;

    /* Sanity check on inputs. */
    if (!baseline_u || !baseline_v || !baseline_w ||
            !station_u || !station_v || !station_w)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Get data type, location and size. */
    type = station_u->type;
    num_stations = station_u->num_elements;
    num_baselines = num_stations * (num_stations - 1) / 2;

    /* Check that the data is in the right location. */
    if (station_u->location != OSKAR_LOCATION_CPU ||
            station_v->location != OSKAR_LOCATION_CPU ||
            station_w->location != OSKAR_LOCATION_CPU ||
            baseline_u->location != OSKAR_LOCATION_CPU ||
            baseline_v->location != OSKAR_LOCATION_CPU ||
            baseline_w->location != OSKAR_LOCATION_CPU)
        return OSKAR_ERR_BAD_LOCATION;

    /* Check that the memory is not NULL. */
    if (!baseline_u->data || !baseline_v->data || !baseline_w->data ||
            !station_u->data || !station_v->data || !station_w->data)
        return OSKAR_ERR_MEMORY_NOT_ALLOCATED;

    /* Check that the data dimensions are OK. */
    if (station_v->num_elements < num_stations ||
            station_w->num_elements < num_stations ||
            baseline_u->num_elements < num_baselines ||
            baseline_v->num_elements < num_baselines ||
            baseline_w->num_elements < num_baselines)
        return OSKAR_ERR_DIMENSION_MISMATCH;

    /* Check that the data is of the right type. */
    if (station_v->type != type || station_w->type != type ||
            baseline_u->type != type || baseline_v->type != type ||
            baseline_w->type != type)
        return OSKAR_ERR_TYPE_MISMATCH;

    if (type == OSKAR_SINGLE)
    {
        oskar_compute_baselines_f(num_stations, (float*)(station_u->data),
                (float*)(station_v->data), (float*)(station_w->data),
                (float*)(baseline_u->data), (float*)(baseline_v->data),
                (float*)(baseline_w->data));
    }
    else if (type == OSKAR_DOUBLE)
    {
        oskar_compute_baselines_d(num_stations, (double*)(station_u->data),
                (double*)(station_v->data), (double*)(station_w->data),
                (double*)(baseline_u->data), (double*)(baseline_v->data),
                (double*)(baseline_w->data));
    }
    else
    {
        return OSKAR_ERR_BAD_DATA_TYPE;
    }

    return error;
}

#ifdef __cplusplus
}
#endif
