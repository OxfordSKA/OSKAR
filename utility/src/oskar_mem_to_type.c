/*
 * Copyright (c) 2013, The University of Oxford
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

#include <oskar_mem_type_check.h>

#ifdef OSKAR_HAVE_CUDA
#include <vector_types.h>
#endif
#include <oskar_vector_types.h>

#ifdef __cplusplus
extern "C" {
#endif

void* oskar_mem_to_void(oskar_Mem* mem)
{
    /* Cast the pointer. */
    return (void*) mem->data;
}

char* oskar_mem_to_char(oskar_Mem* mem, int* status)
{
    /* Check all inputs. */
    if (!mem || !status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Check for type mismatch. */
    if (oskar_mem_base_type(mem->type) != OSKAR_CHAR)
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (char*) mem->data;
}

int* oskar_mem_to_int(oskar_Mem* mem, int* status)
{
    /* Check all inputs. */
    if (!mem || !status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Check for type mismatch. */
    if (oskar_mem_base_type(mem->type) != OSKAR_INT)
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (int*) mem->data;
}

float* oskar_mem_to_float(oskar_Mem* mem, int* status)
{
    /* Check all inputs. */
    if (!mem || !status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Check for type mismatch. */
    if (oskar_mem_base_type(mem->type) != OSKAR_SINGLE)
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (float*) mem->data;
}

float2* oskar_mem_to_float2(oskar_Mem* mem, int* status)
{
    /* Check all inputs. */
    if (!mem || !status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Check for type mismatch. */
    if (oskar_mem_base_type(mem->type) != OSKAR_SINGLE ||
            !oskar_mem_is_complex(mem->type))
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (float2*) mem->data;
}

float4c* oskar_mem_to_float4c(oskar_Mem* mem, int* status)
{
    /* Check all inputs. */
    if (!mem || !status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Check for type mismatch. */
    if (oskar_mem_base_type(mem->type) != OSKAR_SINGLE ||
            !oskar_mem_is_complex(mem->type) ||
            !oskar_mem_is_matrix(mem->type))
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (float4c*) mem->data;
}

double* oskar_mem_to_double(oskar_Mem* mem, int* status)
{
    /* Check all inputs. */
    if (!mem || !status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Check for type mismatch. */
    if (oskar_mem_base_type(mem->type) != OSKAR_DOUBLE)
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (double*) mem->data;
}

double2* oskar_mem_to_double2(oskar_Mem* mem, int* status)
{
    /* Check all inputs. */
    if (!mem || !status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Check for type mismatch. */
    if (oskar_mem_base_type(mem->type) != OSKAR_DOUBLE ||
            !oskar_mem_is_complex(mem->type))
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (double2*) mem->data;
}

double4c* oskar_mem_to_double4c(oskar_Mem* mem, int* status)
{
    /* Check all inputs. */
    if (!mem || !status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Check for type mismatch. */
    if (oskar_mem_base_type(mem->type) != OSKAR_DOUBLE ||
            !oskar_mem_is_complex(mem->type) ||
            !oskar_mem_is_matrix(mem->type))
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (double4c*) mem->data;
}

const void* oskar_mem_to_const_void(const oskar_Mem* mem)
{
    /* Cast the pointer. */
    return (const void*) mem->data;
}

const char* oskar_mem_to_const_char(const oskar_Mem* mem, int* status)
{
    /* Check all inputs. */
    if (!mem || !status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Check for type mismatch. */
    if (oskar_mem_base_type(mem->type) != OSKAR_CHAR)
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (const char*) mem->data;
}

const int* oskar_mem_to_const_int(const oskar_Mem* mem, int* status)
{
    /* Check all inputs. */
    if (!mem || !status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Check for type mismatch. */
    if (oskar_mem_base_type(mem->type) != OSKAR_INT)
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (const int*) mem->data;
}

const float* oskar_mem_to_const_float(const oskar_Mem* mem, int* status)
{
    /* Check all inputs. */
    if (!mem || !status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Check for type mismatch. */
    if (oskar_mem_base_type(mem->type) != OSKAR_SINGLE)
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (const float*) mem->data;
}

const float2* oskar_mem_to_const_float2(const oskar_Mem* mem, int* status)
{
    /* Check all inputs. */
    if (!mem || !status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Check for type mismatch. */
    if (oskar_mem_base_type(mem->type) != OSKAR_SINGLE ||
            !oskar_mem_is_complex(mem->type))
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (const float2*) mem->data;
}

const float4c* oskar_mem_to_const_float4c(const oskar_Mem* mem, int* status)
{
    /* Check all inputs. */
    if (!mem || !status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Check for type mismatch. */
    if (oskar_mem_base_type(mem->type) != OSKAR_SINGLE ||
            !oskar_mem_is_complex(mem->type) ||
            !oskar_mem_is_matrix(mem->type))
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (const float4c*) mem->data;
}

const double* oskar_mem_to_const_double(const oskar_Mem* mem, int* status)
{
    /* Check all inputs. */
    if (!mem || !status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Check for type mismatch. */
    if (oskar_mem_base_type(mem->type) != OSKAR_DOUBLE)
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (const double*) mem->data;
}

const double2* oskar_mem_to_const_double2(const oskar_Mem* mem, int* status)
{
    /* Check all inputs. */
    if (!mem || !status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Check for type mismatch. */
    if (oskar_mem_base_type(mem->type) != OSKAR_DOUBLE ||
            !oskar_mem_is_complex(mem->type))
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (const double2*) mem->data;
}

const double4c* oskar_mem_to_const_double4c(const oskar_Mem* mem, int* status)
{
    /* Check all inputs. */
    if (!mem || !status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Check for type mismatch. */
    if (oskar_mem_base_type(mem->type) != OSKAR_DOUBLE ||
            !oskar_mem_is_complex(mem->type) ||
            !oskar_mem_is_matrix(mem->type))
        *status = OSKAR_ERR_TYPE_MISMATCH;

    /* Cast the pointer. */
    return (const double4c*) mem->data;
}


#ifdef __cplusplus
}
#endif
