/*
 * Copyright (c) 2014-2015, The University of Oxford
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

#ifdef OSKAR_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#include <oskar_mem.h>
#include <private_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

double oskar_mem_get_element_scalar(const oskar_Mem* mem, size_t index,
        int* status)
{
    size_t n;
    int type, location;

    /* Check if safe to proceed. */
    if (*status) return 0.0;

    /* Get the data type, location, and number of elements. */
    type = mem->type;
    location = mem->location;
    n = mem->num_elements;
    if (index >= n)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return 0.0;
    }

    /* Set the data into the array element. */
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_DOUBLE)
        {
            return ((double*)(mem->data))[index];
        }
        else if (type == OSKAR_SINGLE)
        {
            return ((float*)(mem->data))[index];
        }
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (type == OSKAR_DOUBLE)
        {
            double val;
            cudaMemcpy(&val, (double*)(mem->data) + index, sizeof(double),
                    cudaMemcpyDeviceToHost);
            return val;
        }
        else if (type == OSKAR_SINGLE)
        {
            float val;
            cudaMemcpy(&val, (float*)(mem->data) + index, sizeof(float),
                    cudaMemcpyDeviceToHost);
            return val;
        }
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else
        *status = OSKAR_ERR_BAD_LOCATION;
    return 0.0;
}

double2 oskar_mem_get_element_complex(const oskar_Mem* mem, size_t index,
        int* status)
{
    size_t n;
    int type, location;
    double2 val;
    val.x = 0.0;
    val.y = 0.0;

    /* Check if safe to proceed. */
    if (*status) return val;

    /* Get the data type, location, and number of elements. */
    type = mem->type;
    location = mem->location;
    n = mem->num_elements;
    if (index >= n)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return val;
    }

    /* Set the data into the array element. */
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_DOUBLE_COMPLEX)
        {
            return ((double2*)(mem->data))[index];
        }
        else if (type == OSKAR_SINGLE_COMPLEX)
        {
            float2 temp;
            temp = ((float2*)(mem->data))[index];
            val.x = (double) temp.x;
            val.y = (double) temp.y;
        }
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (type == OSKAR_DOUBLE_COMPLEX)
        {
            cudaMemcpy(&val, (double2*)(mem->data) + index, sizeof(double2),
                    cudaMemcpyDeviceToHost);
            return val;
        }
        else if (type == OSKAR_SINGLE_COMPLEX)
        {
            float2 temp;
            cudaMemcpy(&temp, (float2*)(mem->data) + index, sizeof(float2),
                    cudaMemcpyDeviceToHost);
            val.x = (double) temp.x;
            val.y = (double) temp.y;
            return val;
        }
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else
        *status = OSKAR_ERR_BAD_LOCATION;

    return val;
}

double4c oskar_mem_get_element_matrix(const oskar_Mem* mem, size_t index,
        int* status)
{
    size_t n;
    int type, location;
    double4c val;
    val.a.x = 0.0;
    val.a.y = 0.0;
    val.b.x = 0.0;
    val.b.y = 0.0;
    val.c.x = 0.0;
    val.c.y = 0.0;
    val.d.x = 0.0;
    val.d.y = 0.0;

    /* Check if safe to proceed. */
    if (*status) return val;

    /* Get the data type, location, and number of elements. */
    type = mem->type;
    location = mem->location;
    n = mem->num_elements;
    if (index >= n)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return val;
    }

    /* Set the data into the array element. */
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_DOUBLE_COMPLEX_MATRIX)
        {
            return ((double4c*)(mem->data))[index];
        }
        else if (type == OSKAR_SINGLE_COMPLEX_MATRIX)
        {
            float4c temp;
            temp = ((float4c*)(mem->data))[index];
            val.a.x = (double) temp.a.x;
            val.a.y = (double) temp.a.y;
            val.b.x = (double) temp.b.x;
            val.b.y = (double) temp.b.y;
            val.c.x = (double) temp.c.x;
            val.c.y = (double) temp.c.y;
            val.d.x = (double) temp.d.x;
            val.d.y = (double) temp.d.y;
            return val;
        }
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (type == OSKAR_DOUBLE_COMPLEX_MATRIX)
        {
            cudaMemcpy(&val, (double4c*)(mem->data) + index, sizeof(double4c),
                    cudaMemcpyDeviceToHost);
            return val;
        }
        else if (type == OSKAR_SINGLE_COMPLEX_MATRIX)
        {
            float4c temp;
            cudaMemcpy(&temp, (float4c*)(mem->data) + index, sizeof(float4c),
                    cudaMemcpyDeviceToHost);
            val.a.x = (double) temp.a.x;
            val.a.y = (double) temp.a.y;
            val.b.x = (double) temp.b.x;
            val.b.y = (double) temp.b.y;
            val.c.x = (double) temp.c.x;
            val.c.y = (double) temp.c.y;
            val.d.x = (double) temp.d.x;
            val.d.y = (double) temp.d.y;
            return val;
        }
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
    }
    else
        *status = OSKAR_ERR_BAD_LOCATION;

    return val;
}

#ifdef __cplusplus
}
#endif
