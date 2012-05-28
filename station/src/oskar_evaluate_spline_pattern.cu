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

#include "math/oskar_spline_data_evaluate.h"
#include "sky/cudak/oskar_cudak_hor_lmn_to_phi_theta.h"
#include "station/oskar_evaluate_spline_pattern.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_mem_get_pointer.h"
#include "utility/oskar_mem_realloc.h"
#include "utility/oskar_mem_type_check.h"

// Test headers.
#include <cstdio>
#include "station/oskar_element_model_copy.h"
#include "station/oskar_element_model_free.h"
#include "station/oskar_element_model_init.h"
#include "utility/oskar_vector_types.h"

#ifdef __cplusplus
extern "C"
#endif
int oskar_evaluate_spline_pattern(oskar_Mem* pattern,
        const oskar_ElementModel* element, const oskar_Mem* l,
        const oskar_Mem* m, const oskar_Mem* n, double /*cos_orientation_x*/,
        double /*sin_orientation_x*/, double /*cos_orientation_y*/,
        double /*sin_orientation_y*/, oskar_Work* work)
{
    int error, type, num_sources;
    oskar_Mem theta, phi;

    /* Sanity check on inputs. */
    if (!l || !m || !n || !pattern || !work)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Check that all arrays are on the GPU. */
    if (l->location != OSKAR_LOCATION_GPU ||
            m->location != OSKAR_LOCATION_GPU ||
            n->location != OSKAR_LOCATION_GPU ||
            pattern->location != OSKAR_LOCATION_GPU)
        return OSKAR_ERR_BAD_LOCATION;

    /* Check the data type. */
    type = l->type;
    if (type != m->type || type != n->type)
        return OSKAR_ERR_TYPE_MISMATCH;
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
        return OSKAR_ERR_BAD_DATA_TYPE;

    /* Check that the pattern array is a complex matrix. */
    if (!oskar_mem_is_complex(pattern->type) ||
            !oskar_mem_is_matrix(pattern->type))
        return OSKAR_ERR_BAD_DATA_TYPE;

    /* Check that the dimensions are OK. */
    num_sources = l->num_elements;
    if (m->num_elements < num_sources || n->num_elements < num_sources ||
            pattern->num_elements < num_sources)
        return OSKAR_ERR_MEMORY_NOT_ALLOCATED;

    /* FIXME !! This doesn't work for interferometer simulations, since the buffer is already in use for the source horizontal l,m,n values !! */
    /* Ensure enough memory in work buffer to evaluate theta and phi values. */
    if (work->real.num_elements - work->used_real < 2 * num_sources)
    {
        if (work->used_real != 0)
            return OSKAR_ERR_MEMORY_ALLOC_FAILURE; /* Work buffer in use. */
        error = oskar_mem_realloc(&work->real, 2 * num_sources);
        if (error) return error;
    }

    /* Non-owned pointers to the theta and phi work arrays. */
    error = oskar_mem_get_pointer(&theta, &work->real, work->used_real,
            num_sources);
    work->used_real += num_sources;
    if (error) return error;
    error = oskar_mem_get_pointer(&phi, &work->real, work->used_real,
            num_sources);
    work->used_real += num_sources;
    if (error) return error;

    /* Evaluate theta and phi. */
    if (type == OSKAR_SINGLE)
    {
        int num_blocks, num_threads = 256;
        num_blocks = (num_sources + num_threads - 1) / num_threads;
        oskar_cudak_hor_lmn_to_phi_theta_f
        OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_sources,
                (const float*)l->data, (const float*)m->data,
                (const float*)n->data, (float*)phi.data, (float*)theta.data);
    }
    else if (type == OSKAR_DOUBLE)
    {
        int num_blocks, num_threads = 256;
        num_blocks = (num_sources + num_threads - 1) / num_threads;
        oskar_cudak_hor_lmn_to_phi_theta_d
        OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_sources,
                (const double*)l->data, (const double*)m->data,
                (const double*)n->data, (double*)phi.data, (double*)theta.data);
    }

    /* Evaluate the patterns. */
    error = oskar_spline_data_evaluate(pattern, 0, 8, &element->port1_theta_re,
            &theta, &phi);
    if (error) return error;
    error = oskar_spline_data_evaluate(pattern, 1, 8, &element->port1_theta_im,
            &theta, &phi);
    if (error) return error;
    error = oskar_spline_data_evaluate(pattern, 2, 8, &element->port1_phi_re,
            &theta, &phi);
    if (error) return error;
    error = oskar_spline_data_evaluate(pattern, 3, 8, &element->port1_phi_im,
            &theta, &phi);
    if (error) return error;
    error = oskar_spline_data_evaluate(pattern, 4, 8, &element->port2_theta_re,
            &theta, &phi);
    if (error) return error;
    error = oskar_spline_data_evaluate(pattern, 5, 8, &element->port2_theta_im,
            &theta, &phi);
    if (error) return error;
    error = oskar_spline_data_evaluate(pattern, 6, 8, &element->port2_phi_re,
            &theta, &phi);
    if (error) return error;
    error = oskar_spline_data_evaluate(pattern, 7, 8, &element->port2_phi_im,
            &theta, &phi);
    if (error) return error;

    /* Release use of work arrays. */
    work->used_real -= 2 * num_sources;

    /* Report any CUDA error. */
    cudaDeviceSynchronize();
    return (int)cudaPeekAtLastError();
}
