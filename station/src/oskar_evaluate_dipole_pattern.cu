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

#include "oskar_global.h"
#include "station/cudak/oskar_cudak_evaluate_dipole_pattern.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_mem_type_check.h"

#ifdef __cplusplus
extern "C"
#endif
int oskar_evaluate_dipole_pattern(oskar_Mem* pattern, const oskar_Mem* l,
        const oskar_Mem* m, const oskar_Mem* n, double cos_orientation_x,
        double sin_orientation_x, double cos_orientation_y,
        double sin_orientation_y)
{
    int type, num_sources;

    /* Sanity check on inputs. */
    if (!l || !m || !n || !pattern)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Check that all arrays are on the GPU. */
    if (l->location != OSKAR_LOCATION_GPU ||
            m->location != OSKAR_LOCATION_GPU ||
            n->location != OSKAR_LOCATION_GPU ||
            pattern->location != OSKAR_LOCATION_GPU)
        return OSKAR_ERR_BAD_LOCATION;

    /* Check that the pattern array is a complex matrix. */
    if (!oskar_mem_is_complex(pattern->type) ||
            !oskar_mem_is_matrix(pattern->type))
        return OSKAR_ERR_BAD_DATA_TYPE;

    /* Check that the dimensions are OK. */
    num_sources = l->num_elements;
    if (m->num_elements < num_sources || n->num_elements < num_sources ||
            pattern->num_elements < num_sources)
        return OSKAR_ERR_MEMORY_NOT_ALLOCATED;

    /* Switch on the type. */
    type = l->type;
    if (type == OSKAR_SINGLE)
    {
        int num_blocks, num_threads;
        num_threads = 256;
        num_blocks = (num_sources + num_threads - 1) / num_threads;
        oskar_cudak_evaluate_dipole_pattern_f
        OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_sources,
                (const float*)l->data, (const float*)m->data,
                (const float*)n->data, (float)cos_orientation_x,
                (float)sin_orientation_x, (float)cos_orientation_y,
                (float)sin_orientation_y, (float4c*)pattern->data);
    }
    else if (type == OSKAR_DOUBLE)
    {
        int num_blocks, num_threads;
        num_threads = 256;
        num_blocks = (num_sources + num_threads - 1) / num_threads;
        oskar_cudak_evaluate_dipole_pattern_d
        OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_sources,
                (const double*)l->data, (const double*)m->data,
                (const double*)n->data, cos_orientation_x,
                sin_orientation_x, cos_orientation_y, sin_orientation_y,
                (double4c*)pattern->data);
    }
    cudaDeviceSynchronize();
    return (int)cudaPeekAtLastError();
}
