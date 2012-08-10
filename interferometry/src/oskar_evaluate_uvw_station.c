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

#include "interferometry/oskar_evaluate_uvw_station.h"
#include "interferometry/oskar_xyz_to_uvw_cuda.h"
#include "interferometry/oskar_xyz_to_uvw.h"
#include "utility/oskar_cuda_check_error.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_evaluate_uvw_station(oskar_Mem* u, oskar_Mem* v, oskar_Mem* w,
        int num_stations, const oskar_Mem* x, const oskar_Mem* y,
        const oskar_Mem* z, double ra0_rad, double dec0_rad, double gast,
        int* status)
{
    int type, location;
    double ha0_rad;

    /* Check all inputs. */
    if (!u || !v || !w || !x || !y || !z || !status)
    {
        if (status) *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get data type and location of the input coordinates. */
    type = x->type;
    location = x->location;

    /* Check that the memory is not NULL. */
    if (!u->data || !v->data || !w->data || !x->data || !y->data || !z->data)
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;

    /* Check that the data dimensions are OK. */
    if (u->num_elements < num_stations ||
            v->num_elements < num_stations ||
            w->num_elements < num_stations ||
            x->num_elements < num_stations ||
            y->num_elements < num_stations ||
            z->num_elements < num_stations)
        *status = OSKAR_ERR_DIMENSION_MISMATCH;

    /* Check that the data are in the right location. */
    if (y->location != location ||
            z->location != location ||
            u->location != location ||
            v->location != location ||
            w->location != location)
        *status = OSKAR_ERR_BAD_LOCATION;

    /* Check that the data is of the right type. */
    if (y->type != type || z->type != type ||
            u->type != type || v->type != type || w->type != type)
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Evaluate Greenwich Hour Angle of phase centre. */
    ha0_rad = gast - ra0_rad;

    /* Evaluate station u,v,w coordinates. */
    if (location == OSKAR_LOCATION_GPU)
    {
        if (type == OSKAR_SINGLE)
        {
            oskar_xyz_to_uvw_cuda_f(num_stations,
                    (float*)(x->data), (float*)(y->data), (float*)(z->data),
                    (float)ha0_rad, (float)dec0_rad,
                    (float*)(u->data), (float*)(v->data), (float*)(w->data));
        }
        else if (type == OSKAR_DOUBLE)
        {
            oskar_xyz_to_uvw_cuda_d(num_stations,
                    (double*)(x->data), (double*)(y->data), (double*)(z->data),
                    ha0_rad, dec0_rad,
                    (double*)(u->data), (double*)(v->data), (double*)(w->data));
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
        oskar_cuda_check_error(status);
    }
    else if (location == OSKAR_LOCATION_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            oskar_xyz_to_uvw_f(num_stations,
                    (float*)(x->data), (float*)(y->data), (float*)(z->data),
                    (float)ha0_rad, (float)dec0_rad,
                    (float*)(u->data), (float*)(v->data), (float*)(w->data));
        }
        else if (type == OSKAR_DOUBLE)
        {
            oskar_xyz_to_uvw_d(num_stations,
                    (double*)(x->data), (double*)(y->data), (double*)(z->data),
                    ha0_rad, dec0_rad,
                    (double*)(u->data), (double*)(v->data), (double*)(w->data));
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
    else
    {
        *status = OSKAR_ERR_BAD_LOCATION;
    }
}

#ifdef __cplusplus
}
#endif
