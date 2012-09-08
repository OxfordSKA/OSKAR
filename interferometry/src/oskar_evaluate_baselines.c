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
#include "interferometry/oskar_evaluate_baselines_cuda.h"
#include "utility/oskar_cuda_check_error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_evaluate_baselines_f(float* uu, float* vv, float* ww,
        int num_stations, const float* u, const float* v, const float* w)
{
    int s1, s2, b; /* Station and baseline indices. */
    for (s1 = 0, b = 0; s1 < num_stations; ++s1)
    {
        for (s2 = s1 + 1; s2 < num_stations; ++s2, ++b)
        {
            uu[b] = u[s2] - u[s1];
            vv[b] = v[s2] - v[s1];
            ww[b] = w[s2] - w[s1];
        }
    }
}

/* Double precision. */
void oskar_evaluate_baselines_d(double* uu, double* vv, double* ww,
        int num_stations, const double* u, const double* v, const double* w)
{
    int s1, s2, b; /* Station and baseline indices. */
    for (s1 = 0, b = 0; s1 < num_stations; ++s1)
    {
        for (s2 = s1 + 1; s2 < num_stations; ++s2, ++b)
        {
            uu[b] = u[s2] - u[s1];
            vv[b] = v[s2] - v[s1];
            ww[b] = w[s2] - w[s1];
        }
    }
}

/* Wrapper. */
void oskar_evaluate_baselines(oskar_Mem* uu, oskar_Mem* vv, oskar_Mem* ww,
        const oskar_Mem* u, const oskar_Mem* v, const oskar_Mem* w,
        int* status)
{
    int type, location, num_stations, num_baselines;

    /* Check all inputs. */
    if (!uu || !vv || !ww || !u || !v || !w || !status)
    {
        if (status) *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get data type, location and size. */
    type = u->type;
    location = u->location;
    num_stations = u->num_elements;
    num_baselines = num_stations * (num_stations - 1) / 2;

    /* Check that the data types match. */
    if (v->type != type || w->type != type ||
            uu->type != type || vv->type != type || ww->type != type)
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Check that the data locations match. */
    if (v->location != location ||
            w->location != location ||
            uu->location != location ||
            vv->location != location ||
            ww->location != location)
        *status = OSKAR_ERR_BAD_LOCATION;

    /* Check that the memory is not NULL. */
    if (!uu->data || !vv->data || !ww->data ||
            !u->data || !v->data || !w->data)
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;

    /* Check that the data dimensions are OK. */
    if (v->num_elements < num_stations ||
            w->num_elements < num_stations ||
            uu->num_elements < num_baselines ||
            vv->num_elements < num_baselines ||
            ww->num_elements < num_baselines)
        *status = OSKAR_ERR_DIMENSION_MISMATCH;

    /* Check if safe to proceed. */
    if (*status) return;

    if (location == OSKAR_LOCATION_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            oskar_evaluate_baselines_f((float*)(uu->data), (float*)(vv->data),
                    (float*)(ww->data), num_stations, (float*)(u->data),
                    (float*)(v->data), (float*)(w->data));
        }
        else if (type == OSKAR_DOUBLE)
        {
            oskar_evaluate_baselines_d((double*)(uu->data), (double*)(vv->data),
                    (double*)(ww->data), num_stations, (double*)(u->data),
                    (double*)(v->data), (double*)(w->data));
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
    else if (location == OSKAR_LOCATION_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (type == OSKAR_SINGLE)
        {
            oskar_evaluate_baselines_cuda_f((float*)(uu->data),
                    (float*)(vv->data), (float*)(ww->data), num_stations,
                    (float*)(u->data), (float*)(v->data), (float*)(w->data));
            oskar_cuda_check_error(status);
        }
        else if (type == OSKAR_DOUBLE)
        {
            oskar_evaluate_baselines_cuda_d((double*)(uu->data),
                    (double*)(vv->data), (double*)(ww->data), num_stations,
                    (double*)(u->data), (double*)(v->data), (double*)(w->data));
            oskar_cuda_check_error(status);
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
}

#ifdef __cplusplus
}
#endif
