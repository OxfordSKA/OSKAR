/*
 * Copyright (c) 2011-2014, The University of Oxford
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

#include <private_mem.h>
#include <oskar_mem.h>
#include <oskar_mem_add_cuda.h>
#include <oskar_cuda_check_error.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_add(oskar_Mem* a, const oskar_Mem* b, const oskar_Mem* c,
        int* status)
{
    int type, location;
    size_t i, num_elements;

    /* Check all inputs. */
    if (!a || !b || !c || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get meta-data. */
    location = oskar_mem_location(a);
    type = oskar_mem_type(a);
    num_elements = oskar_mem_length(a);

    /* Check for empty array. */
    if (num_elements == 0)
        return;

    /* Check data types, locations, and number of elements. */
    if (type != oskar_mem_type(b) || type != oskar_mem_type(c))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (location != oskar_mem_location(b) || location != oskar_mem_location(c))
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }
    if (num_elements != oskar_mem_length(b) ||
            num_elements != oskar_mem_length(c))
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Get total number of elements. */
    if (oskar_mem_type_is_matrix(type))
        num_elements *= 4;
    if (oskar_mem_type_is_complex(type))
        num_elements *= 2;

    /* Switch on type and location. */
    if (oskar_mem_type_is_double(type))
    {
        double *aa;
        const double *bb, *cc;
        aa = oskar_mem_double(a, status);
        bb = oskar_mem_double_const(b, status);
        cc = oskar_mem_double_const(c, status);

        if (location == OSKAR_CPU)
        {
            for (i = 0; i < num_elements; ++i)
                aa[i] = bb[i] + cc[i];
        }
        else if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_mem_add_cuda_d(num_elements, bb, cc, aa);
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else
            *status = OSKAR_ERR_BAD_LOCATION;
    }
    else if (oskar_mem_type_is_single(type))
    {
        float *aa;
        const float *bb, *cc;
        aa = oskar_mem_float(a, status);
        bb = oskar_mem_float_const(b, status);
        cc = oskar_mem_float_const(c, status);

        if (location == OSKAR_CPU)
        {
            for (i = 0; i < num_elements; ++i)
                aa[i] = bb[i] + cc[i];
        }
        else if (location == OSKAR_GPU)
        {
#ifdef OSKAR_HAVE_CUDA
            oskar_mem_add_cuda_f(num_elements, bb, cc, aa);
            oskar_cuda_check_error(status);
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
        else
            *status = OSKAR_ERR_BAD_LOCATION;
    }
    else
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
}

#ifdef __cplusplus
}
#endif
