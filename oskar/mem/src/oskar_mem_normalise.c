/*
 * Copyright (c) 2019, The University of Oxford
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

#include "mem/define_mem_normalise.h"
#include "mem/oskar_mem.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_MEM_NORMALISE_REAL_CPU(    M_CAT(mem_norm_real_, float), float)
OSKAR_MEM_NORMALISE_COMPLEX_CPU( M_CAT(mem_norm_complex_, float), float, float2)
OSKAR_MEM_NORMALISE_MATRIX_CPU(  M_CAT(mem_norm_matrix_, float), float, float4c)
OSKAR_MEM_NORMALISE_REAL_CPU(    M_CAT(mem_norm_real_, double), double)
OSKAR_MEM_NORMALISE_COMPLEX_CPU( M_CAT(mem_norm_complex_, double), double, double2)
OSKAR_MEM_NORMALISE_MATRIX_CPU(  M_CAT(mem_norm_matrix_, double), double, double4c)

/*
 * For matrix types, return square root of autocorrelation:
 * sqrt(0.5 * [sum of resultant diagonal]).
 *
 * We have
 * [ Xa  Xb ] [ Xa*  Xc* ] = [ Xa Xa* + Xb Xb*    (don't care)   ]
 * [ Xc  Xd ] [ Xb*  Xd* ]   [  (don't care)     Xc Xc* + Xd Xd* ]
 *
 * Stokes I is completely real, so need only evaluate the real
 * part of all the multiplies. Because of the conjugate terms,
 * these become re*re + im*im.
 *
 * Need the square root because we only want the normalised value
 * for the beam itself (in isolation), not its actual
 * autocorrelation!
 */

void oskar_mem_normalise(oskar_Mem* mem, size_t offset,
        size_t num_elements, size_t norm_index, int* status)
{
    if (*status) return;
    const int type = oskar_mem_type(mem);
    const int location = oskar_mem_location(mem);
    const unsigned int off = (unsigned int) offset;
    const unsigned int n = (unsigned int) num_elements;
    const unsigned int idx = (unsigned int) norm_index;
    if (location == OSKAR_CPU)
    {
        switch (type)
        {
        case OSKAR_DOUBLE:
            mem_norm_real_double(off, n,
                    oskar_mem_double(mem, status), idx);
            break;
        case OSKAR_DOUBLE_COMPLEX:
            mem_norm_complex_double(off, n,
                    oskar_mem_double2(mem, status), idx);
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            mem_norm_matrix_double(off, n,
                    oskar_mem_double4c(mem, status), idx);
            break;
        case OSKAR_SINGLE:
            mem_norm_real_float(off, n,
                    oskar_mem_float(mem, status), idx);
            break;
        case OSKAR_SINGLE_COMPLEX:
            mem_norm_complex_float(off, n,
                    oskar_mem_float2(mem, status), idx);
            break;
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            mem_norm_matrix_float(off, n,
                    oskar_mem_float4c(mem, status), idx);
            break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        switch (type)
        {
        case OSKAR_DOUBLE:
            k = "mem_norm_real_double";
            break;
        case OSKAR_DOUBLE_COMPLEX:
            k = "mem_norm_complex_double";
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            k = "mem_norm_matrix_double";
            break;
        case OSKAR_SINGLE:
            k = "mem_norm_real_float";
            break;
        case OSKAR_SINGLE_COMPLEX:
            k = "mem_norm_complex_float";
            break;
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            k = "mem_norm_matrix_float";
            break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(num_elements, local_size[0]);
        const oskar_Arg args[] = {
                {INT_SZ, &off},
                {INT_SZ, &n},
                {PTR_SZ, oskar_mem_buffer(mem)},
                {INT_SZ, &idx}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
