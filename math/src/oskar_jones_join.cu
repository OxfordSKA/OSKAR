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
#include "math/cudak/oskar_cudak_jones_mul_c2.h"
#include "math/cudak/oskar_cudak_jones_mul_mat1_c1.h"
#include "math/cudak/oskar_cudak_jones_mul_mat2.h"
#include "math/cudak/oskar_cudak_jones_mul_scalar_c2.h"

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
int oskar_jones_copy_from_gpu(int type, int n_sources, int n_stations,
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
int oskar_jones_copy_to_gpu(int type, int n_sources, int n_stations,
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
    int n_sources1 = 0, n_stations1 = 0;
    int n_sources2 = 0, n_stations2 = 0;
    int n_sources3 = 0, n_stations3 = 0;
#ifdef __cplusplus
    n_sources1 = j1->n_sources();
    n_sources2 = j2->n_sources();
    n_sources3 = j3->n_sources();
    n_stations1 = j1->n_stations();
    n_stations2 = j2->n_stations();
    n_stations3 = j3->n_stations();
#else
    n_sources1 = j1->private_n_sources;
    n_sources2 = j2->private_n_sources;
    n_sources3 = j3->private_n_sources;
    n_stations1 = j1->private_n_stations;
    n_stations2 = j2->private_n_stations;
    n_stations3 = j3->private_n_stations;
#endif

    // Check the data dimensions.
    if (n_sources1 != n_sources2 || n_sources1 != n_sources3)
        return -11;
    if (n_stations1 != n_stations2 || n_stations1 != n_stations3)
        return -12;

    // Figure out what we've been given.
    int type1 = 0, location1 = 0;
    int type2 = 0, location2 = 0;
    int type3 = 0, location3 = 0;
#ifdef __cplusplus
    type1 = j1->type();
    type2 = j2->type();
    type3 = j3->type();
    location1 = j1->location();
    location2 = j2->location();
    location3 = j3->location();
#else
    type1 = j1->private_type;
    type2 = j2->private_type;
    type3 = j3->private_type;
    location1 = j1->private_location;
    location2 = j2->private_location;
    location3 = j3->private_location;
#endif

    // Check that there is enough memory to hold the result.
    size_t size1 = oskar_jones_element_size_from_type(type1);
    size_t size2 = oskar_jones_element_size_from_type(type2);
    size_t size3 = oskar_jones_element_size_from_type(type3);
    if (size3 < size2 || size3 < size1)
        return -20;

    // Set up CUDA thread block sizes.
    int n_thd = 0, n_blk = 0;
    int n_elements = n_sources1 * n_stations1;

    // Can no longer guarantee a clean return from this point,
    // so wait until end in case of errors.
    void *d_data1 = 0, *d_data2 = 0, *d_data3 = 0;
    int err_code = 0;

    // Set up GPU pointer to J1.
    if (location1 == 0)
    {
        err_code = oskar_jones_alloc_gpu(type1, n_sources1, n_stations1,
                &d_data1);
        err_code = oskar_jones_copy_to_gpu(type1, n_sources1, n_stations1,
                j1->data, d_data1);
    }
    else d_data1 = j1->data;

    // Set up GPU pointer to J2.
    if (location2 == 0)
    {
        err_code = oskar_jones_alloc_gpu(type2, n_sources2, n_stations2,
                &d_data2);
        err_code = oskar_jones_copy_to_gpu(type2, n_sources2, n_stations2,
                j2->data, d_data2);
    }
    else d_data2 = j2->data;

    // Set up GPU pointer to J3.
    if (location3 == 0)
    {
        err_code = oskar_jones_alloc_gpu(type3, n_sources3, n_stations3,
                &d_data3);
    }
    else d_data3 = j3->data;

    // Check if errors occurred.
    if (err_code != 0) goto stop;

    // Set error code to type mismatch by default.
    err_code = -100;

    // Multiply the matrices.
    if (type1 == OSKAR_JONES_FLOAT_SCALAR)
    {
        if (type2 == OSKAR_JONES_FLOAT_SCALAR)
        {
            if (type3 == OSKAR_JONES_FLOAT_SCALAR)
            {
                // Scalar-scalar to scalar, float: OK.
                n_thd = 256;
                n_blk = (n_elements + n_thd - 1) / n_thd;
                oskar_cudak_jones_mul_scalar_c2_f
                OSKAR_CUDAK_CONF(n_blk, n_thd) (
                        n_elements, (const float2*)d_data1,
                        (const float2*)d_data2, (float2*)d_data3);
                cudaDeviceSynchronize();
                err_code = cudaPeekAtLastError();
            }
            else if (type3 == OSKAR_JONES_FLOAT_MATRIX)
            {
                // Scalar-scalar to matrix, float: OK.
                n_thd = 256;
                n_blk = (n_elements + n_thd - 1) / n_thd;
                oskar_cudak_jones_mul_c2_f
                OSKAR_CUDAK_CONF(n_blk, n_thd) (
                        n_elements, (const float2*)d_data1,
                        (const float2*)d_data2, (float4c*)d_data3);
                cudaDeviceSynchronize();
                err_code = cudaPeekAtLastError();
            }
        }
        else if (type2 == OSKAR_JONES_FLOAT_MATRIX)
        {
            // Scalar-matrix, float: OK.
            n_thd = 256;
            n_blk = (n_elements + n_thd - 1) / n_thd;
            oskar_cudak_jones_mul_mat1_c1_f
            OSKAR_CUDAK_CONF(n_blk, n_thd) (
                    n_elements, (const float4c*)d_data2,
                    (const float2*)d_data1, (float4c*)d_data3);
            cudaDeviceSynchronize();
            err_code = cudaPeekAtLastError();
        }
    }
    else if (type1 == OSKAR_JONES_FLOAT_MATRIX)
    {
        if (type2 == OSKAR_JONES_FLOAT_SCALAR)
        {
            // Matrix-scalar, float: OK.
            n_thd = 256;
            n_blk = (n_elements + n_thd - 1) / n_thd;
            oskar_cudak_jones_mul_mat1_c1_f
            OSKAR_CUDAK_CONF(n_blk, n_thd) (
                    n_elements, (const float4c*)d_data1,
                    (const float2*)d_data2, (float4c*)d_data3);
            cudaDeviceSynchronize();
            err_code = cudaPeekAtLastError();
        }
        else if (type2 == OSKAR_JONES_FLOAT_MATRIX)
        {
            // Matrix-matrix, float: OK.
            n_thd = 256;
            n_blk = (n_elements + n_thd - 1) / n_thd;
            oskar_cudak_jones_mul_mat2_f
            OSKAR_CUDAK_CONF(n_blk, n_thd) (
                    n_elements, (const float4c*)d_data1,
                    (const float4c*)d_data2, (float4c*)d_data3);
            cudaDeviceSynchronize();
            err_code = cudaPeekAtLastError();
        }
    }
    else if (type1 == OSKAR_JONES_DOUBLE_SCALAR)
    {
        if (type2 == OSKAR_JONES_DOUBLE_SCALAR)
        {
            if (type3 == OSKAR_JONES_DOUBLE_SCALAR)
            {
                // Scalar-scalar to scalar, double: OK.
                n_thd = 256;
                n_blk = (n_elements + n_thd - 1) / n_thd;
                oskar_cudak_jones_mul_scalar_c2_d
                OSKAR_CUDAK_CONF(n_blk, n_thd) (
                        n_elements, (const double2*)d_data1,
                        (const double2*)d_data2, (double2*)d_data3);
                cudaDeviceSynchronize();
                err_code = cudaPeekAtLastError();
            }
            else if (type3 == OSKAR_JONES_DOUBLE_MATRIX)
            {
                // Scalar-scalar to matrix, double: OK.
                n_thd = 256;
                n_blk = (n_elements + n_thd - 1) / n_thd;
                oskar_cudak_jones_mul_c2_d
                OSKAR_CUDAK_CONF(n_blk, n_thd) (
                        n_elements, (const double2*)d_data1,
                        (const double2*)d_data2, (double4c*)d_data3);
                cudaDeviceSynchronize();
                err_code = cudaPeekAtLastError();
            }
        }
        else if (type2 == OSKAR_JONES_DOUBLE_MATRIX)
        {
            // Scalar-matrix, double: OK.
            n_thd = 256;
            n_blk = (n_elements + n_thd - 1) / n_thd;
            oskar_cudak_jones_mul_mat1_c1_d
            OSKAR_CUDAK_CONF(n_blk, n_thd) (
                    n_elements, (const double4c*)d_data2,
                    (const double2*)d_data1, (double4c*)d_data3);
            cudaDeviceSynchronize();
            err_code = cudaPeekAtLastError();
        }
    }
    else if (type1 == OSKAR_JONES_DOUBLE_MATRIX)
    {
        if (type2 == OSKAR_JONES_DOUBLE_SCALAR)
        {
            // Matrix-scalar, double: OK.
            n_thd = 256;
            n_blk = (n_elements + n_thd - 1) / n_thd;
            oskar_cudak_jones_mul_mat1_c1_d
            OSKAR_CUDAK_CONF(n_blk, n_thd) (
                    n_elements, (const double4c*)d_data1,
                    (const double2*)d_data2, (double4c*)d_data3);
            cudaDeviceSynchronize();
            err_code = cudaPeekAtLastError();
        }
        else if (type2 == OSKAR_JONES_DOUBLE_MATRIX)
        {
            // Matrix-matrix, double: OK.
            n_thd = 256;
            n_blk = (n_elements + n_thd - 1) / n_thd;
            oskar_cudak_jones_mul_mat2_d
            OSKAR_CUDAK_CONF(n_blk, n_thd) (
                    n_elements, (const double4c*)d_data1,
                    (const double4c*)d_data2, (double4c*)d_data3);
            cudaDeviceSynchronize();
            err_code = cudaPeekAtLastError();
        }
    }

stop:
    // Free GPU memory if input data was on the host.
    if (location1 == 0) cudaFree(d_data1);
    if (location2 == 0) cudaFree(d_data2);

    // Copy result back to host memory if required.
    if (location3 == 0)
    {
        if (err_code == 0)
        {
            err_code = oskar_jones_copy_from_gpu(type3,
                    n_sources3,	n_stations3, d_data3, j3->data);
        }
        cudaFree(d_data3);
    }
    return err_code;
}
