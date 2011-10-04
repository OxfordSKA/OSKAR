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

#include "math/oskar_jones_join.h"
#include "math/oskar_cuda_jones_mul_c2.h"
#include "math/oskar_cuda_jones_mul_mat1_c1.h"
#include "math/oskar_cuda_jones_mul_mat2.h"
#include "math/oskar_cuda_jones_mul_scalar_c2.h"

#ifdef __cplusplus
extern "C"
#endif
size_t oskar_jones_element_size_from_type(int type)
{
    if (type == OSKAR_JONES_FLOAT_SCALAR)
        return sizeof(float2);
    else if (type == OSKAR_JONES_DOUBLE_SCALAR)
        return sizeof(double2);
    else if (type == OSKAR_JONES_FLOAT_MATRIX)
        return sizeof(float4c);
    else if (type == OSKAR_JONES_DOUBLE_MATRIX)
        return sizeof(double4c);
    return 0;
}

#ifdef __cplusplus
extern "C"
#endif
int oskar_jones_alloc(oskar_Jones* jones)
{
    // Check that the structure exists.
    if (jones == NULL) return -1;

    // Get the meta-data.
#ifdef __cplusplus
    int n_sources = jones->n_sources();
    int n_stations = jones->n_stations();
    int location = jones->location();
    int type = jones->type();
#else
    int n_sources = jones->private_n_sources;
    int n_stations = jones->private_n_stations;
    int location = jones->private_location;
    int type = jones->private_type;
#endif

    // Get the memory size.
    size_t element_size = oskar_jones_element_size_from_type(type);
    size_t bytes = n_sources * n_stations * element_size;

    // Check whether the memory should be on the host or the device.
    int err = 0;
    if (location == 0)
    {
        // Allocate host memory.
        jones->data = malloc(bytes);
        if (jones->data == NULL) err = -1;
    }
    else if (location == 1)
    {
        // Allocate GPU memory.
        cudaMalloc(&jones->data, bytes);
        err = cudaPeekAtLastError();
    }
    return err;
}

#ifdef __cplusplus
extern "C"
#endif
int oskar_jones_free(oskar_Jones* jones)
{
    // Check that the structure exists.
    if (jones == NULL) return -1;

    // Get the meta-data.
#ifdef __cplusplus
    int location = jones->location();
#else
    int location = jones->private_location;
#endif

    // Check whether the memory is on the host or the device.
    int err = 0;
    if (location == 0)
    {
        // Free host memory.
        free(jones->data);
    }
    else if (location == 1)
    {
        // Free GPU memory.
        cudaFree(jones->data);
        err = cudaPeekAtLastError();
    }
    jones->data = NULL;
    return err;
}

#ifdef __cplusplus
extern "C"
#endif
int oskar_jones_alloc_gpu(int type, int n_sources, int n_stations,
        void** d_data)
{
    // Allocate GPU memory.
    size_t element_size = oskar_jones_element_size_from_type(type);
    cudaMalloc(d_data, element_size * n_sources * n_stations);
    return cudaPeekAtLastError();
}

#ifdef __cplusplus
extern "C"
#endif
int oskar_jones_copy_memory_from_gpu(int type, int n_sources, int n_stations,
        const void* d_data, void* h_data)
{
    // Copy the data across.
    size_t element_size = oskar_jones_element_size_from_type(type);
    cudaMemcpy(h_data, d_data, element_size * n_sources * n_stations,
            cudaMemcpyDeviceToHost);
    return cudaPeekAtLastError();
}

#ifdef __cplusplus
extern "C"
#endif
int oskar_jones_copy_memory_to_gpu(int type, int n_sources, int n_stations,
        const void* h_data, void* d_data)
{
    // Copy the data across.
    size_t element_size = oskar_jones_element_size_from_type(type);
    cudaMemcpy(d_data, h_data, element_size * n_sources * n_stations,
            cudaMemcpyHostToDevice);
    return cudaPeekAtLastError();
}

#ifdef __cplusplus
extern "C"
#endif
int oskar_jones_join(oskar_Jones* j3, oskar_Jones* j1, const oskar_Jones* j2)
{
    // Check to see if the output structure J3 exists: if not, set it to J1.
    if (j3 == NULL) j3 = j1;

    // Check that all pointers are not NULL.
    if (j1 == NULL) return -1;
    if (j2 == NULL) return -2;

    // Check that the memory in each structure is allocated.
    if (j1->data == NULL) return -1;
    if (j2->data == NULL) return -2;
    if (j3->data == NULL) return -3;

    // Get the dimensions of the input data.
#ifdef __cplusplus
    int n_sources1 = j1->n_sources();
    int n_sources2 = j2->n_sources();
    int n_sources3 = j3->n_sources();
    int n_stations1 = j1->n_stations();
    int n_stations2 = j2->n_stations();
    int n_stations3 = j3->n_stations();
#else
    int n_sources1 = j1->private_n_sources;
    int n_sources2 = j2->private_n_sources;
    int n_sources3 = j3->private_n_sources;
    int n_stations1 = j1->private_n_stations;
    int n_stations2 = j2->private_n_stations;
    int n_stations3 = j3->private_n_stations;
#endif

    // Check the data dimensions.
    if (n_sources1 != n_sources2 || n_sources1 != n_sources3)
        return -11;
    if (n_stations1 != n_stations2 || n_stations1 != n_stations3)
        return -12;

    // Figure out what we've been given.
#ifdef __cplusplus
    int type1 = j1->type();
    int type2 = j2->type();
    int type3 = j3->type();
    int location1 = j1->location();
    int location2 = j2->location();
    int location3 = j3->location();
#else
    int type1 = j1->private_type;
    int type2 = j2->private_type;
    int type3 = j3->private_type;
    int location1 = j1->private_location;
    int location2 = j2->private_location;
    int location3 = j3->private_location;
#endif

    // Check that there is enough memory to hold the result.
    size_t size1 = oskar_jones_element_size_from_type(type1);
    size_t size2 = oskar_jones_element_size_from_type(type2);
    size_t size3 = oskar_jones_element_size_from_type(type3);
    if (size3 < size2 || size3 < size1)
        return -20;

    // Set up CUDA thread block sizes.
    int n_elements = n_sources1 * n_stations1;

    // Can no longer guarantee a clean return from this point,
    // so wait until end in case of errors.
    void *d1 = 0, *d2 = 0, *d3 = 0;
    int err = 0;

    // Set up GPU pointer to J1.
    if (location1 == 0)
    {
        err = oskar_jones_alloc_gpu(type1, n_sources1, n_stations1, &d1);
        err = oskar_jones_copy_memory_to_gpu(type1, n_sources1, n_stations1,
                j1->data, d1);
    }
    else d1 = j1->data;

    // Set up GPU pointer to J2.
    if (location2 == 0)
    {
        err = oskar_jones_alloc_gpu(type2, n_sources2, n_stations2, &d2);
        err = oskar_jones_copy_memory_to_gpu(type2, n_sources2, n_stations2,
                j2->data, d2);
    }
    else d2 = j2->data;

    // Set up GPU pointer to J3.
    if (location3 == 0)
        err = oskar_jones_alloc_gpu(type3, n_sources3, n_stations3, &d3);
    else d3 = j3->data;

    // Check if errors occurred.
    if (err != 0) goto stop;

    // Set error code to type mismatch by default.
    err = -100;

    // Multiply the matrices.
    if (type1 == OSKAR_JONES_FLOAT_SCALAR)
    {
        if (type2 == OSKAR_JONES_FLOAT_SCALAR)
        {
            if (type3 == OSKAR_JONES_FLOAT_SCALAR)
            {
                // Scalar-scalar to scalar, float: OK.
                err = oskar_cuda_jones_mul_scalar_c2_f(n_elements,
                        (const float2*)d1, (const float2*)d2, (float2*)d3);
            }
            else if (type3 == OSKAR_JONES_FLOAT_MATRIX)
            {
                // Scalar-scalar to matrix, float: OK.
                err = oskar_cuda_jones_mul_c2_f(n_elements,
                        (const float2*)d1, (const float2*)d2, (float4c*)d3);
            }
        }
        else if (type2 == OSKAR_JONES_FLOAT_MATRIX)
        {
            // Scalar-matrix, float: OK.
            err = oskar_cuda_jones_mul_mat1_c1_f(n_elements,
                    (const float4c*)d2, (const float2*)d1, (float4c*)d3);
        }
    }
    else if (type1 == OSKAR_JONES_FLOAT_MATRIX)
    {
        if (type2 == OSKAR_JONES_FLOAT_SCALAR)
        {
            // Matrix-scalar, float: OK.
            err = oskar_cuda_jones_mul_mat1_c1_f(n_elements,
                    (const float4c*)d1, (const float2*)d2, (float4c*)d3);
        }
        else if (type2 == OSKAR_JONES_FLOAT_MATRIX)
        {
            // Matrix-matrix, float: OK.
            err = oskar_cuda_jones_mul_mat2_f(n_elements,
                    (const float4c*)d1, (const float4c*)d2, (float4c*)d3);
        }
    }
    else if (type1 == OSKAR_JONES_DOUBLE_SCALAR)
    {
        if (type2 == OSKAR_JONES_DOUBLE_SCALAR)
        {
            if (type3 == OSKAR_JONES_DOUBLE_SCALAR)
            {
                // Scalar-scalar to scalar, double: OK.
                err = oskar_cuda_jones_mul_scalar_c2_d(n_elements,
                        (const double2*)d1, (const double2*)d2, (double2*)d3);
            }
            else if (type3 == OSKAR_JONES_DOUBLE_MATRIX)
            {
                // Scalar-scalar to matrix, double: OK.
                err = oskar_cuda_jones_mul_c2_d(n_elements,
                        (const double2*)d1, (const double2*)d2, (double4c*)d3);
            }
        }
        else if (type2 == OSKAR_JONES_DOUBLE_MATRIX)
        {
            // Scalar-matrix, double: OK.
            err = oskar_cuda_jones_mul_mat1_c1_d(n_elements,
                    (const double4c*)d2, (const double2*)d1, (double4c*)d3);
        }
    }
    else if (type1 == OSKAR_JONES_DOUBLE_MATRIX)
    {
        if (type2 == OSKAR_JONES_DOUBLE_SCALAR)
        {
            // Matrix-scalar, double: OK.
            err = oskar_cuda_jones_mul_mat1_c1_d(n_elements,
                    (const double4c*)d1, (const double2*)d2, (double4c*)d3);
        }
        else if (type2 == OSKAR_JONES_DOUBLE_MATRIX)
        {
            // Matrix-matrix, double: OK.
            err = oskar_cuda_jones_mul_mat2_d (n_elements,
                    (const double4c*)d1, (const double4c*)d2, (double4c*)d3);
        }
    }

stop:
    // Free GPU memory if input data was on the host.
    if (location1 == 0) cudaFree(d1);
    if (location2 == 0) cudaFree(d2);

    // Copy result back to host memory if required.
    if (location3 == 0)
    {
        if (err == 0)
        {
            err = oskar_jones_copy_memory_from_gpu(type3,
                    n_sources3,	n_stations3, d3, j3->data);
        }
        cudaFree(d3);
    }
    return err;
}
