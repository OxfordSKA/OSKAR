/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#include "interferometry/oskar_evaluate_jones_K.h"
#include "interferometry/oskar_evaluate_jones_K_cuda.h"
#include "utility/oskar_cuda_check_error.h"
#include "utility/oskar_mem_type_check.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_evaluate_jones_K_f(float2* jones, int num_stations,
        const float* u, const float* v, const float* w, int num_sources,
        const float* l, const float* m, const float* n)
{
    int station, source;

    /* Loop over stations. */
    for (station = 0; station < num_stations; ++station)
    {
        float us, vs, ws;
        float2* station_ptr;

        /* Get the station data. */
        station_ptr = &jones[station * num_sources];
        us = u[station];
        vs = v[station];
        ws = w[station];

        /* Loop over sources. */
        for (source = 0; source < num_sources; ++source)
        {
            float phase;
            float2 weight;

            /* Calculate the source phase. */
            phase = us * l[source] + vs * m[source] + ws * n[source];
            weight.x = cosf(phase);
            weight.y = sinf(phase);

            /* Store the result. */
            station_ptr[source] = weight;
        }
    }
}

/* Double precision. */
void oskar_evaluate_jones_K_d(double2* jones, int num_stations,
        const double* u, const double* v, const double* w, int num_sources,
        const double* l, const double* m, const double* n)
{
    int station, source;

    /* Loop over stations. */
    for (station = 0; station < num_stations; ++station)
    {
        double us, vs, ws;
        double2* station_ptr;

        /* Get the station data. */
        station_ptr = &jones[station * num_sources];
        us = u[station];
        vs = v[station];
        ws = w[station];

        /* Loop over sources. */
        for (source = 0; source < num_sources; ++source)
        {
            double phase;
            double2 weight;

            /* Calculate the source phase. */
            phase = us * l[source] + vs * m[source] + ws * n[source];
            weight.x = cos(phase);
            weight.y = sin(phase);

            /* Store the result. */
            station_ptr[source] = weight;
        }
    }
}

/* Wrapper. */
void oskar_evaluate_jones_K(oskar_Jones* K, const oskar_Mem* l,
        const oskar_Mem* m, const oskar_Mem* n, const oskar_Mem* u,
        const oskar_Mem* v, const oskar_Mem* w, int* status)
{
    int num_sources, num_stations, jones_type, base_type, location;

    /* Check all inputs. */
    if (!K || !l || !m || !n || !u || !v || !w || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the Jones matrix block meta-data. */
    jones_type = K->data.type;
    base_type = oskar_mem_base_type(jones_type);
    location = K->data.location;
    num_sources = K->num_sources;
    num_stations = K->num_stations;

    /* Check that the memory is not NULL. */
    if (!K->data.data || !l->data || !m->data || !n->data ||
            !u->data || !v->data || !w->data)
    {
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;
        return;
    }

    /* Check that the data dimensions are OK. */
    if (num_sources > l->num_elements ||
            num_sources > m->num_elements ||
            num_sources > n->num_elements ||
            num_stations != u->num_elements ||
            num_stations != v->num_elements ||
            num_stations != w->num_elements)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Check that the data is in the right location. */
    if (l->location != location || m->location != location ||
            n->location != location || u->location != location ||
            v->location != location || w->location != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Check that the data are of the right type. */
    if (!oskar_mem_is_complex(jones_type) || oskar_mem_is_matrix(jones_type))
    {
        *status = OSKAR_ERR_BAD_JONES_TYPE;
        return;
    }
    if (base_type != l->type || base_type != m->type ||
            base_type != n->type || base_type != u->type ||
            base_type != v->type || base_type != w->type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Evaluate Jones matrices. */
    if (location == OSKAR_LOCATION_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (jones_type == OSKAR_SINGLE_COMPLEX)
        {
            oskar_evaluate_jones_K_cuda_f((float2*)(K->data.data),
                    num_stations,
                    (const float*)(u->data),
                    (const float*)(v->data),
                    (const float*)(w->data),
                    num_sources,
                    (const float*)(l->data),
                    (const float*)(m->data),
                    (const float*)(n->data));
        }
        else if (jones_type == OSKAR_DOUBLE_COMPLEX)
        {
            oskar_evaluate_jones_K_cuda_d((double2*)(K->data.data),
                    num_stations,
                    (const double*)(u->data),
                    (const double*)(v->data),
                    (const double*)(w->data),
                    num_sources,
                    (const double*)(l->data),
                    (const double*)(m->data),
                    (const double*)(n->data));
        }
        oskar_cuda_check_error(status);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else if (location == OSKAR_LOCATION_CPU)
    {
        if (jones_type == OSKAR_SINGLE_COMPLEX)
        {
            oskar_evaluate_jones_K_f((float2*)(K->data.data),
                    num_stations,
                    (const float*)(u->data),
                    (const float*)(v->data),
                    (const float*)(w->data),
                    num_sources,
                    (const float*)(l->data),
                    (const float*)(m->data),
                    (const float*)(n->data));
        }
        else if (jones_type == OSKAR_DOUBLE_COMPLEX)
        {
            oskar_evaluate_jones_K_d((double2*)(K->data.data),
                    num_stations,
                    (const double*)(u->data),
                    (const double*)(v->data),
                    (const double*)(w->data),
                    num_sources,
                    (const double*)(l->data),
                    (const double*)(m->data),
                    (const double*)(n->data));
        }
    }
}

#ifdef __cplusplus
}
#endif
