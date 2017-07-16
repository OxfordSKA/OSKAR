/*
 * Copyright (c) 2013-2017, The University of Oxford
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

#ifdef __cplusplus
extern "C" {
#endif

int oskar_mem_allocated(const oskar_Mem* mem)
{
    return mem->data ? 1 : 0;
}

#ifdef OSKAR_HAVE_OPENCL
cl_mem* oskar_mem_cl_buffer(oskar_Mem* mem, int* status)
{
    if (mem->location & OSKAR_CL)
        return &mem->buffer;
    else
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return 0;
    }
}

const cl_mem* oskar_mem_cl_buffer_const(const oskar_Mem* mem, int* status)
{
    if (mem->location & OSKAR_CL)
        return &mem->buffer;
    else
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return 0;
    }
}
#endif

size_t oskar_mem_length(const oskar_Mem* mem)
{
    return mem->num_elements;
}

int oskar_mem_location(const oskar_Mem* mem)
{
    return mem->location;
}

int oskar_mem_type(const oskar_Mem* mem)
{
    return mem->type;
}

int oskar_mem_precision(const oskar_Mem* mem)
{
    return oskar_type_precision(mem->type);
}

int oskar_mem_is_double(const oskar_Mem* mem)
{
    return oskar_type_is_double(mem->type);
}

int oskar_mem_is_single(const oskar_Mem* mem)
{
    return oskar_type_is_single(mem->type);
}

int oskar_mem_is_complex(const oskar_Mem* mem)
{
    return oskar_type_is_complex(mem->type);
}

int oskar_mem_is_real(const oskar_Mem* mem)
{
    return oskar_type_is_real(mem->type);
}

int oskar_mem_is_matrix(const oskar_Mem* mem)
{
    return oskar_type_is_matrix(mem->type);
}

int oskar_mem_is_scalar(const oskar_Mem* mem)
{
    return oskar_type_is_scalar(mem->type);
}

/* Pointer conversion functions. */

void* oskar_mem_void(oskar_Mem* mem)
{
    return (void*) mem->data;
}

const void* oskar_mem_void_const(const oskar_Mem* mem)
{
    return (const void*) mem->data;
}

char* oskar_mem_char(oskar_Mem* mem)
{
    return (char*) mem->data;
}

const char* oskar_mem_char_const(const oskar_Mem* mem)
{
    return (const char*) mem->data;
}

int* oskar_mem_int(oskar_Mem* mem, int* status)
{
    /* Check for type mismatch. */
    if (oskar_type_precision(mem->type) != OSKAR_INT)
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (int*) mem->data;
}

const int* oskar_mem_int_const(const oskar_Mem* mem, int* status)
{
    /* Check for type mismatch. */
    if (oskar_type_precision(mem->type) != OSKAR_INT)
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (const int*) mem->data;
}

float* oskar_mem_float(oskar_Mem* mem, int* status)
{
    /* Check for type mismatch. */
    if (oskar_type_precision(mem->type) != OSKAR_SINGLE)
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (float*) mem->data;
}

const float* oskar_mem_float_const(const oskar_Mem* mem, int* status)
{
    /* Check for type mismatch. */
    if (oskar_type_precision(mem->type) != OSKAR_SINGLE)
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (const float*) mem->data;
}

float2* oskar_mem_float2(oskar_Mem* mem, int* status)
{
    /* Check for type mismatch. */
    if (oskar_type_precision(mem->type) != OSKAR_SINGLE ||
            !oskar_type_is_complex(mem->type))
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (float2*) mem->data;
}

const float2* oskar_mem_float2_const(const oskar_Mem* mem, int* status)
{
    /* Check for type mismatch. */
    if (oskar_type_precision(mem->type) != OSKAR_SINGLE ||
            !oskar_type_is_complex(mem->type))
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (const float2*) mem->data;
}

float4c* oskar_mem_float4c(oskar_Mem* mem, int* status)
{
    /* Check for type mismatch. */
    if (oskar_type_precision(mem->type) != OSKAR_SINGLE ||
            !oskar_type_is_complex(mem->type) ||
            !oskar_type_is_matrix(mem->type))
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (float4c*) mem->data;
}

const float4c* oskar_mem_float4c_const(const oskar_Mem* mem, int* status)
{
    /* Check for type mismatch. */
    if (oskar_type_precision(mem->type) != OSKAR_SINGLE ||
            !oskar_type_is_complex(mem->type) ||
            !oskar_type_is_matrix(mem->type))
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (const float4c*) mem->data;
}

double* oskar_mem_double(oskar_Mem* mem, int* status)
{
    /* Check for type mismatch. */
    if (oskar_type_precision(mem->type) != OSKAR_DOUBLE)
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (double*) mem->data;
}

const double* oskar_mem_double_const(const oskar_Mem* mem, int* status)
{
    /* Check for type mismatch. */
    if (oskar_type_precision(mem->type) != OSKAR_DOUBLE)
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (const double*) mem->data;
}

double2* oskar_mem_double2(oskar_Mem* mem, int* status)
{
    /* Check for type mismatch. */
    if (oskar_type_precision(mem->type) != OSKAR_DOUBLE ||
            !oskar_type_is_complex(mem->type))
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (double2*) mem->data;
}

const double2* oskar_mem_double2_const(const oskar_Mem* mem, int* status)
{
    /* Check for type mismatch. */
    if (oskar_type_precision(mem->type) != OSKAR_DOUBLE ||
            !oskar_type_is_complex(mem->type))
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (const double2*) mem->data;
}

double4c* oskar_mem_double4c(oskar_Mem* mem, int* status)
{
    /* Check for type mismatch. */
    if (oskar_type_precision(mem->type) != OSKAR_DOUBLE ||
            !oskar_type_is_complex(mem->type) ||
            !oskar_type_is_matrix(mem->type))
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (double4c*) mem->data;
}

const double4c* oskar_mem_double4c_const(const oskar_Mem* mem, int* status)
{
    /* Check for type mismatch. */
    if (oskar_type_precision(mem->type) != OSKAR_DOUBLE ||
            !oskar_type_is_complex(mem->type) ||
            !oskar_type_is_matrix(mem->type))
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (const double4c*) mem->data;
}


#ifdef __cplusplus
}
#endif
