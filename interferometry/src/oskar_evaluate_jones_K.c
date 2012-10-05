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

#include "interferometry/oskar_evaluate_jones_K.h"
#include "interferometry/oskar_evaluate_jones_K_cuda.h"
#include "utility/oskar_cuda_check_error.h"
#include "utility/oskar_mem_type_check.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Wrapper. */
void oskar_evaluate_jones_K(oskar_Jones* K, const oskar_SkyModel* sky,
        const oskar_Mem* u, const oskar_Mem* v, const oskar_Mem* w,
        int* status)
{
    int num_sources, num_stations, jones_type, base_type, location;

    /* Check all inputs. */
    if (!K || !sky || !u || !v || !w || !status)
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
    if (!K->data.data || !sky->rel_l.data || !sky->rel_m.data ||
            !sky->rel_n.data || !u->data || !v->data || !w->data)
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;

    /* Check that the data dimensions are OK. */
    if (K->num_sources != sky->num_sources ||
            K->num_stations != u->num_elements ||
            K->num_stations != v->num_elements ||
            K->num_stations != w->num_elements)
        *status = OSKAR_ERR_DIMENSION_MISMATCH;

    /* Check that the data is in the right location. */
    if (sky->rel_l.location != location || sky->rel_m.location != location ||
            sky->rel_n.location != location || u->location != location ||
            v->location != location || w->location != location)
        *status = OSKAR_ERR_BAD_LOCATION;

    /* Check that the data are of the right type. */
    if (!oskar_mem_is_complex(jones_type) || oskar_mem_is_matrix(jones_type))
        *status = OSKAR_ERR_BAD_JONES_TYPE;
    if (base_type != sky->rel_l.type ||
            base_type != sky->rel_m.type ||
            base_type != sky->rel_n.type ||
            base_type != u->type ||
            base_type != v->type ||
            base_type != w->type)
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Check if safe to proceed. */
    if (*status) return;

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
                    (const float*)(sky->rel_l.data),
                    (const float*)(sky->rel_m.data),
                    (const float*)(sky->rel_n.data));
        }
        else if (jones_type == OSKAR_DOUBLE_COMPLEX)
        {
            oskar_evaluate_jones_K_cuda_d((double2*)(K->data.data),
                    num_stations,
                    (const double*)(u->data),
                    (const double*)(v->data),
                    (const double*)(w->data),
                    num_sources,
                    (const double*)(sky->rel_l.data),
                    (const double*)(sky->rel_m.data),
                    (const double*)(sky->rel_n.data));
        }
        oskar_cuda_check_error(status);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else if (location == OSKAR_LOCATION_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
    }
}

#ifdef __cplusplus
}
#endif
