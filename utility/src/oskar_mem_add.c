/*
 * Copyright (c) 2011-2015, The University of Oxford
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

#include <oskar_mem.h>
#include <oskar_mem_add_cuda.h>
#include <oskar_cuda_check_error.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_add(oskar_Mem* out, const oskar_Mem* in1, const oskar_Mem* in2,
        size_t num_elements, int* status)
{
    int type, precision, location;
    size_t i;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get meta-data. */
    type = oskar_mem_type(out);
    precision = oskar_mem_precision(out);
    location = oskar_mem_location(out);

    /* Check for empty array. */
    if (oskar_mem_length(in1) == 0)
        return;

    /* Check data types, locations, and number of elements. */
    if (type != oskar_mem_type(in1) || type != oskar_mem_type(in2))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (location != oskar_mem_location(in1) ||
            location != oskar_mem_location(in2))
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }
    if (oskar_mem_length(out) < num_elements ||
            oskar_mem_length(in1) < num_elements ||
            oskar_mem_length(in2) < num_elements)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Get total number of elements to add. */
    if (oskar_mem_is_matrix(out))
        num_elements *= 4;
    if (oskar_mem_is_complex(out))
        num_elements *= 2;

    /* Switch on type and location. */
    if (precision == OSKAR_DOUBLE)
    {
        double *aa;
        const double *bb, *cc;
        aa = oskar_mem_double(out, status);
        bb = oskar_mem_double_const(in1, status);
        cc = oskar_mem_double_const(in2, status);

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
    else if (precision == OSKAR_SINGLE)
    {
        float *aa;
        const float *bb, *cc;
        aa = oskar_mem_float(out, status);
        bb = oskar_mem_float_const(in1, status);
        cc = oskar_mem_float_const(in2, status);

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
