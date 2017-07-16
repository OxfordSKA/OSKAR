/*
 * Copyright (c) 2013-2015, The University of Oxford
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

#include "mem/oskar_mem.h"
#include "mem/private_mem.h"

#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SDF "% .6e "
#define SDD "% .14e "

void oskar_mem_save_ascii(FILE* file, size_t num_mem, size_t num_elements,
        int* status, ...)
{
    int type;
    size_t i, j;
    va_list args;
    oskar_Mem** handles; /* Array of oskar_Mem pointers in CPU memory. */

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check there are at least the number of specified elements in
     * each array. */
    va_start(args, status);
    for (i = 0; i < num_mem; ++i)
    {
        const oskar_Mem* mem;
        mem = va_arg(args, const oskar_Mem*);
        if (oskar_mem_length(mem) < num_elements)
            *status = OSKAR_ERR_DIMENSION_MISMATCH;
    }
    va_end(args);

    /* Check if safe to proceed. */
    if (*status) return;

    /* Allocate and set up the handle array. */
    handles = (oskar_Mem**) malloc(num_mem * sizeof(oskar_Mem*));
    va_start(args, status);
    for (i = 0; i < num_mem; ++i)
    {
        oskar_Mem* mem;
        mem = va_arg(args, oskar_Mem*);
        if (oskar_mem_location(mem) != OSKAR_CPU)
        {
            handles[i] = oskar_mem_create_copy(mem, OSKAR_CPU, status);
        }
        else
        {
            handles[i] = mem;
        }
    }
    va_end(args);

    for (j = 0; j < num_elements; ++j)
    {
        /* Break if error. */
        if (*status) break;

        for (i = 0; i < num_mem; ++i)
        {
            const void* data;
            data = oskar_mem_void_const(handles[i]);
            type = oskar_mem_type(handles[i]);
            switch (type)
            {
            case OSKAR_SINGLE:
            {
                fprintf(file, SDF, ((const float*)data)[j]);
                continue;
            }
            case OSKAR_DOUBLE:
            {
                fprintf(file, SDD, ((const double*)data)[j]);
                continue;
            }
            case OSKAR_SINGLE_COMPLEX:
            {
                float2 d;
                d = ((const float2*)data)[j];
                fprintf(file, SDF SDF, d.x, d.y);
                continue;
            }
            case OSKAR_DOUBLE_COMPLEX:
            {
                double2 d;
                d = ((const double2*)data)[j];
                fprintf(file, SDD SDD, d.x, d.y);
                continue;
            }
            case OSKAR_SINGLE_COMPLEX_MATRIX:
            {
                float4c d;
                d = ((const float4c*)data)[j];
                fprintf(file, SDF SDF SDF SDF SDF SDF SDF SDF,
                        d.a.x, d.a.y, d.b.x, d.b.y, d.c.x, d.c.y, d.d.x, d.d.y);
                continue;
            }
            case OSKAR_DOUBLE_COMPLEX_MATRIX:
            {
                double4c d;
                d = ((const double4c*)data)[j];
                fprintf(file, SDD SDD SDD SDD SDD SDD SDD SDD,
                        d.a.x, d.a.y, d.b.x, d.b.y, d.c.x, d.c.y, d.d.x, d.d.y);
                continue;
            }
            case OSKAR_CHAR:
            {
                putc(((const char*)data)[j], file);
                continue;
            }
            case OSKAR_INT:
            {
                fprintf(file, "%5d ", ((const int*)data)[j]);
                continue;
            }
            default:
            {
                *status = OSKAR_ERR_BAD_DATA_TYPE;
                continue;
            }
            }
        }
        putc('\n', file);
    }

    /* Free any temporary memory used by this function. */
    va_start(args, status);
    for (i = 0; i < num_mem; ++i)
    {
        const oskar_Mem* mem;
        mem = va_arg(args, const oskar_Mem*);
        if (oskar_mem_location(mem) != OSKAR_CPU)
        {
            oskar_mem_free(handles[i], status);
        }
    }
    va_end(args);

    /* Free the handle array. */
    free(handles);
}

#ifdef __cplusplus
}
#endif
