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

#include <cuda_runtime_api.h>
#include "math/oskar_jones_join.h"
#include "math/oskar_jones_element_size.h"
#include "math/oskar_cuda_jones_mul_c2.h"
#include "math/oskar_cuda_jones_mul_mat1_c1.h"
#include "math/oskar_cuda_jones_mul_mat2.h"
#include "math/oskar_cuda_jones_mul_scalar_c2.h"

extern "C"
int oskar_jones_join(oskar_Jones* j3, oskar_Jones* j1, const oskar_Jones* j2)
{
    // Check to see if the output structure J3 exists: if not, set it to J1.
    if (j3 == NULL) j3 = j1;

    // Check that all pointers are not NULL.
    if (j1 == NULL) return -1;
    if (j2 == NULL) return -2;
    if (j1->data == NULL) return -1;
    if (j2->data == NULL) return -2;
    if (j3->data == NULL) return -3;

    // Get the dimensions of the input data.
    int n_sources1 = j1->n_sources();
    int n_sources2 = j2->n_sources();
    int n_sources3 = j3->n_sources();
    int n_stations1 = j1->n_stations();
    int n_stations2 = j2->n_stations();
    int n_stations3 = j3->n_stations();

    // Check the data dimensions.
    if (n_sources1 != n_sources2 || n_sources1 != n_sources3)
        return -11;
    if (n_stations1 != n_stations2 || n_stations1 != n_stations3)
        return -12;

    // Figure out what we've been given.
    int type1 = j1->type();
    int type2 = j2->type();
    int type3 = j3->type();
    int location1 = j1->location();
    int location2 = j2->location();
    int location3 = j3->location();

    // Check that there is enough memory to hold the result.
    size_t size1 = oskar_jones_element_size(type1);
    size_t size2 = oskar_jones_element_size(type2);
    size_t size3 = oskar_jones_element_size(type3);
    if (size3 < size2 || size3 < size1)
        return -20;

    // Copy data to GPU if required.
    int n_elements = n_sources1 * n_stations1;
    const oskar_Jones* hd1 = (location1 == 0) ? new oskar_Jones(j1, 1) : j1;
    const oskar_Jones* hd2 = (location2 == 0) ? new oskar_Jones(j2, 1) : j2;
    oskar_Jones* hd3 = (location3 == 0) ? new oskar_Jones(j3, 1) : j3;
    const void* d1 = hd1->data;
    const void* d2 = hd2->data;
    void* d3 = hd3->data;

    // Check for errors.
    int err = cudaPeekAtLastError();
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
    if (location1 == 0) delete hd1;
    if (location2 == 0) delete hd2;

    // Copy result back to host memory if required.
    if (location3 == 0)
    {
        if (err == 0)
            err = hd3->copy_to(j3);
        delete hd3;
    }
    return err;
}
