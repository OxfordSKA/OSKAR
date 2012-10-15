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

#include "station/oskar_evaluate_dipole_pattern_cuda.h"
#include "utility/oskar_mem_type_check.h"
#include "utility/oskar_cuda_check_error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Wrapper. */
void oskar_evaluate_dipole_pattern(oskar_Mem* pattern, int num_points,
        const oskar_Mem* theta, const oskar_Mem* phi, int return_x_dipole,
        int* status)
{
    int type, location;

    /* Check all inputs. */
    if (!theta || !phi || !pattern || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the meta-data. */
    location = pattern->location;
    type = theta->type;

    /* Check that all arrays are co-located. */
    if (theta->location != location || phi->location != location)
        *status = OSKAR_ERR_BAD_LOCATION;

    /* Check that the pattern array is a complex matrix. */
    if (!oskar_mem_is_complex(pattern->type) ||
            !oskar_mem_is_matrix(pattern->type))
        *status = OSKAR_ERR_BAD_DATA_TYPE;

    /* Check that the dimensions are OK. */
    if (theta->num_elements < num_points || phi->num_elements < num_points ||
            pattern->num_elements < num_points)
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check the location. */
    if (location == OSKAR_LOCATION_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (type == OSKAR_SINGLE)
        {
            oskar_evaluate_dipole_pattern_cuda_f(num_points,
                    (const float*)theta->data, (const float*)phi->data,
                    return_x_dipole, (float4c*)pattern->data);
        }
        else if (type == OSKAR_DOUBLE)
        {
            oskar_evaluate_dipole_pattern_cuda_d(num_points,
                    (const double*)theta->data, (const double*)phi->data,
                    return_x_dipole, (double4c*)pattern->data);
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
