/*
 * Copyright (c) 2012, The University of Oxford
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
#include <vector_functions.h>

#include "utility/test/Test_Mem.h"
#include "utility/oskar_binary_file_read.h"
#include "utility/oskar_binary_file_write.h"
#include "utility/oskar_binary_tag_index_free.h"
#include "utility/oskar_BinaryTag.h"
#include "utility/oskar_mem_add.h"
#include "utility/oskar_mem_add_gaussian_noise.h"
#include "utility/oskar_mem_append.h"
#include "utility/oskar_mem_append_raw.h"
#include "utility/oskar_mem_binary_file_read.h"
#include "utility/oskar_mem_binary_file_write.h"
#include "utility/oskar_mem_copy.h"
#include "utility/oskar_mem_different.h"
#include "utility/oskar_mem_realloc.h"
#include "utility/oskar_mem_set_value_real.h"
#include "utility/oskar_mem_scale_real.h"
#include "utility/oskar_mem_type_check.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_file_exists.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_vector_types.h"
#include "utility/oskar_Settings.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>


using namespace std;

void Test_Mem::test_alloc()
{
}

void Test_Mem::test_realloc()
{
    int error = 0;

    oskar_Mem mem_gpu(OSKAR_DOUBLE, OSKAR_LOCATION_GPU, 0);
    oskar_mem_realloc(&mem_gpu, 500, &error);
    CPPUNIT_ASSERT_EQUAL_MESSAGE((error > 0) ? std::string("CUDA ERROR: ") +
            cudaGetErrorString((cudaError_t)error) : "OSKAR ERROR", 0, error);
    CPPUNIT_ASSERT_EQUAL(500, mem_gpu.num_elements);
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_DOUBLE, mem_gpu.type);

    oskar_Mem mem_cpu(OSKAR_DOUBLE_COMPLEX, OSKAR_LOCATION_CPU, 100);
    oskar_mem_realloc(&mem_cpu, 1000, &error);
    CPPUNIT_ASSERT_EQUAL_MESSAGE((error > 0) ? std::string("CUDA ERROR: ") +
            cudaGetErrorString((cudaError_t)error) : "OSKAR ERROR", 0, error);
    CPPUNIT_ASSERT_EQUAL(1000, mem_cpu.num_elements);
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_DOUBLE_COMPLEX, mem_cpu.type);
}


void Test_Mem::test_append()
{
    int status = 0;
    {
        oskar_Mem mem_cpu(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, 0);
        int num_values1 = 10;
        double value1 = 1.0;
        vector<double> data1(num_values1, value1);
        oskar_mem_append_raw(&mem_cpu, (const void*)&data1[0], OSKAR_DOUBLE,
                OSKAR_LOCATION_CPU, num_values1, &status);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
        CPPUNIT_ASSERT_EQUAL(num_values1, mem_cpu.num_elements);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_CPU, mem_cpu.location);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_DOUBLE, mem_cpu.type);
        for (int i = 0; i < mem_cpu.num_elements; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(value1, ((double*)mem_cpu.data)[i], 1.0e-5);
        }
        int num_values2 = 5;
        double value2 = 2.0;
        vector<double> data2(num_values2, value2);
        oskar_mem_append_raw(&mem_cpu, (const void*)&data2[0], OSKAR_DOUBLE,
                OSKAR_LOCATION_CPU, num_values2, &status);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
        CPPUNIT_ASSERT_EQUAL(num_values1 + num_values2, mem_cpu.num_elements);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_CPU, mem_cpu.location);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_DOUBLE, mem_cpu.type);
        for (int i = 0; i < mem_cpu.num_elements; ++i)
        {
            if (i < num_values1)
                CPPUNIT_ASSERT_DOUBLES_EQUAL(value1, ((double*)mem_cpu.data)[i], 1.0e-5);
            else
                CPPUNIT_ASSERT_DOUBLES_EQUAL(value2, ((double*)mem_cpu.data)[i], 1.0e-5);
        }
    }

    {
        oskar_Mem mem_gpu(OSKAR_SINGLE, OSKAR_LOCATION_GPU, 0);
        int num_values1 = 10;
        float value1 = 1.0;
        vector<float> data1(num_values1, value1);
        oskar_mem_append_raw(&mem_gpu, (const void*)&data1[0],
                OSKAR_SINGLE, OSKAR_LOCATION_CPU, num_values1, &status);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
        CPPUNIT_ASSERT_EQUAL(num_values1, mem_gpu.num_elements);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_GPU, mem_gpu.location);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, mem_gpu.type);
        oskar_Mem mem_temp(&mem_gpu, OSKAR_LOCATION_CPU);
        for (int i = 0; i < mem_gpu.num_elements; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(value1, ((float*)mem_temp.data)[i], 1.0e-5);
        }

        int num_values2 = 5;
        float value2 = 2.0;
        vector<float> data2(num_values2, value2);
        oskar_mem_append_raw(&mem_gpu, (const void*)&data2[0],
                OSKAR_SINGLE, OSKAR_LOCATION_CPU, num_values2, &status);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);
        CPPUNIT_ASSERT_EQUAL(num_values1 + num_values2, mem_gpu.num_elements);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_GPU, mem_gpu.location);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, mem_gpu.type);
        oskar_Mem mem_temp2(&mem_gpu, OSKAR_LOCATION_CPU);
        for (int i = 0; i < mem_gpu.num_elements; ++i)
        {
            if (i < num_values1)
                CPPUNIT_ASSERT_DOUBLES_EQUAL(value1, ((float*)mem_temp2.data)[i], 1.0e-5);
            else
                CPPUNIT_ASSERT_DOUBLES_EQUAL(value2, ((float*)mem_temp2.data)[i], 1.0e-5);
        }
    }
}

void Test_Mem::test_different()
{
    int error = 0, value;

    // Test two memory blocks that are the same.
    {
        oskar_Mem one(OSKAR_SINGLE, OSKAR_LOCATION_CPU, 20);
        oskar_Mem two(OSKAR_SINGLE, OSKAR_LOCATION_CPU, 20);

        oskar_mem_set_value_real(&one, 4.4, &error);
        oskar_mem_set_value_real(&two, 4.4, &error);
        CPPUNIT_ASSERT_EQUAL(0, error);

        value = oskar_mem_different(&one, &two, 0, &error);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_FALSE, value);
        CPPUNIT_ASSERT_EQUAL(0, error);
    }

    // Test two memory blocks that are different.
    {
        oskar_Mem one(OSKAR_SINGLE, OSKAR_LOCATION_CPU, 20);
        oskar_Mem two(OSKAR_SINGLE, OSKAR_LOCATION_CPU, 20);

        oskar_mem_set_value_real(&one, 4.4, &error);
        oskar_mem_set_value_real(&two, 4.2, &error);
        CPPUNIT_ASSERT_EQUAL(0, error);

        value = oskar_mem_different(&one, &two, 0, &error);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_TRUE, value);
        CPPUNIT_ASSERT_EQUAL(0, error);
    }

    // Test two memory blocks that are different by one element.
    {
        oskar_Mem one(OSKAR_SINGLE, OSKAR_LOCATION_CPU, 20);
        oskar_Mem two(OSKAR_SINGLE, OSKAR_LOCATION_CPU, 20);

        oskar_mem_set_value_real(&one, 1.0, &error);
        oskar_mem_set_value_real(&two, 1.0, &error);
        CPPUNIT_ASSERT_EQUAL(0, error);
        ((float*)(two.data))[4] = 1.1;

        value = oskar_mem_different(&one, &two, 0, &error);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_TRUE, value);
        CPPUNIT_ASSERT_EQUAL(0, error);
    }

    // Test two memory blocks that are different by one element, but only up to
    // the point where they are different.
    {
        oskar_Mem one(OSKAR_SINGLE, OSKAR_LOCATION_CPU, 20);
        oskar_Mem two(OSKAR_SINGLE, OSKAR_LOCATION_CPU, 20);

        oskar_mem_set_value_real(&one, 1.0, &error);
        oskar_mem_set_value_real(&two, 1.0, &error);
        CPPUNIT_ASSERT_EQUAL(0, error);
        ((float*)(two.data))[4] = 1.1;

        value = oskar_mem_different(&one, &two, 4, &error);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_FALSE, value);
        CPPUNIT_ASSERT_EQUAL(0, error);
    }
}

void Test_Mem::test_type_check()
{
    {
        oskar_Mem mem(OSKAR_SINGLE, OSKAR_LOCATION_CPU, 0);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_FALSE, mem.is_double());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_FALSE, mem.is_complex());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_TRUE, mem.is_scalar());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_FALSE, oskar_mem_is_double(OSKAR_SINGLE));
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_FALSE, oskar_mem_is_complex(OSKAR_SINGLE));
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_TRUE, oskar_mem_is_scalar(OSKAR_SINGLE));
    }

    {
        oskar_Mem mem(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, 0);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_TRUE, mem.is_double());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_FALSE, mem.is_complex());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_TRUE, mem.is_scalar());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_TRUE, oskar_mem_is_double(OSKAR_DOUBLE));
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_FALSE, oskar_mem_is_complex(OSKAR_DOUBLE));
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_TRUE, oskar_mem_is_scalar(OSKAR_DOUBLE));
    }

    {
        oskar_Mem mem(OSKAR_SINGLE_COMPLEX, OSKAR_LOCATION_CPU, 0);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_FALSE, mem.is_double());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_TRUE, mem.is_complex());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_TRUE, mem.is_scalar());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_FALSE, oskar_mem_is_double(OSKAR_SINGLE_COMPLEX));
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_TRUE, oskar_mem_is_complex(OSKAR_SINGLE_COMPLEX));
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_TRUE, oskar_mem_is_scalar(OSKAR_SINGLE_COMPLEX));
    }

    {
        oskar_Mem mem(OSKAR_DOUBLE_COMPLEX, OSKAR_LOCATION_CPU, 0);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_TRUE, mem.is_double());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_TRUE, mem.is_complex());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_TRUE, mem.is_scalar());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_TRUE, oskar_mem_is_double(OSKAR_DOUBLE_COMPLEX));
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_TRUE, oskar_mem_is_complex(OSKAR_DOUBLE_COMPLEX));
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_TRUE, oskar_mem_is_scalar(OSKAR_DOUBLE_COMPLEX));

    }

    {
        oskar_Mem mem(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_LOCATION_CPU, 0);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_FALSE, mem.is_double());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_TRUE, mem.is_complex());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_FALSE, mem.is_scalar());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_FALSE, oskar_mem_is_double(OSKAR_SINGLE_COMPLEX_MATRIX));
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_TRUE, oskar_mem_is_complex(OSKAR_SINGLE_COMPLEX_MATRIX));
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_FALSE, oskar_mem_is_scalar(OSKAR_SINGLE_COMPLEX_MATRIX));
    }

    {
        oskar_Mem mem(OSKAR_DOUBLE_COMPLEX_MATRIX, OSKAR_LOCATION_CPU, 0);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_TRUE, mem.is_double());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_TRUE, mem.is_complex());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_FALSE, mem.is_scalar());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_TRUE, oskar_mem_is_double(OSKAR_DOUBLE_COMPLEX_MATRIX));
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_TRUE, oskar_mem_is_complex(OSKAR_DOUBLE_COMPLEX_MATRIX));
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_FALSE, oskar_mem_is_scalar(OSKAR_DOUBLE_COMPLEX_MATRIX));
    }
}

void Test_Mem::test_scale_real()
{
    int n = 100;
    int status = 0;

    // Single precision real.
    {
        oskar_Mem mem_cpu(OSKAR_SINGLE, OSKAR_LOCATION_CPU, n);

        // Fill memory.
        for (int i = 0; i < n; ++i)
        {
            ((float*)(mem_cpu.data))[i] = (float)i;
        }

        // Scale.
        oskar_mem_scale_real(&mem_cpu, 2.0, &status);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);

        // Check contents.
        for (int i = 0; i < n; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0f * i,
                    ((float*)(mem_cpu.data))[i], 1e-6);
        }

        // Copy to GPU.
        oskar_Mem mem_gpu(&mem_cpu, OSKAR_LOCATION_GPU);

        // Scale again.
        oskar_mem_scale_real(&mem_gpu, 2.0, &status);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);

        // Copy back and check contents.
        oskar_Mem mem_cpu2(&mem_gpu, OSKAR_LOCATION_CPU);
        for (int i = 0; i < n; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0f * i,
                    ((float*)(mem_cpu2.data))[i], 1e-6);
        }
    }

    // Single precision complex.
    {
        oskar_Mem mem_cpu(OSKAR_SINGLE_COMPLEX, OSKAR_LOCATION_CPU, n);

        // Fill memory.
        for (int i = 0; i < n; ++i)
        {
            ((float2*)(mem_cpu.data))[i].x = (float)i;
            ((float2*)(mem_cpu.data))[i].y = (float)i + 0.2f;
        }

        // Scale.
        oskar_mem_scale_real(&mem_cpu, 2.0, &status);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);

        // Check contents.
        for (int i = 0; i < n; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0f * ((float)i),
                    ((float2*)(mem_cpu.data))[i].x, 1e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0f * ((float)i + 0.2f),
                    ((float2*)(mem_cpu.data))[i].y, 1e-6);
        }

        // Copy to GPU.
        oskar_Mem mem_gpu(&mem_cpu, OSKAR_LOCATION_GPU);

        // Scale again.
        oskar_mem_scale_real(&mem_gpu, 2.0, &status);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);

        // Copy back and check contents.
        oskar_Mem mem_cpu2(&mem_gpu, OSKAR_LOCATION_CPU);
        for (int i = 0; i < n; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0f * ((float)i),
                    ((float2*)(mem_cpu2.data))[i].x, 1e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0f * ((float)i + 0.2f),
                    ((float2*)(mem_cpu2.data))[i].y, 1e-6);
        }
    }

    // Single precision complex matrix.
    {
        oskar_Mem mem_cpu(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_LOCATION_CPU, n);

        // Fill memory.
        for (int i = 0; i < n; ++i)
        {
            ((float4c*)(mem_cpu.data))[i].a.x = (float)i;
            ((float4c*)(mem_cpu.data))[i].a.y = (float)i + 0.2f;
            ((float4c*)(mem_cpu.data))[i].b.x = (float)i + 0.4f;
            ((float4c*)(mem_cpu.data))[i].b.y = (float)i + 0.6f;
            ((float4c*)(mem_cpu.data))[i].c.x = (float)i + 0.8f;
            ((float4c*)(mem_cpu.data))[i].c.y = (float)i + 1.0f;
            ((float4c*)(mem_cpu.data))[i].d.x = (float)i + 1.2f;
            ((float4c*)(mem_cpu.data))[i].d.y = (float)i + 1.4f;
        }

        // Scale.
        oskar_mem_scale_real(&mem_cpu, 2.0, &status);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);

        // Check contents.
        for (int i = 0; i < n; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0f * ((float)i),
                    ((float4c*)(mem_cpu.data))[i].a.x, 1e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0f * ((float)i + 0.2f),
                    ((float4c*)(mem_cpu.data))[i].a.y, 1e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0f * ((float)i + 0.4f),
                    ((float4c*)(mem_cpu.data))[i].b.x, 1e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0f * ((float)i + 0.6f),
                    ((float4c*)(mem_cpu.data))[i].b.y, 1e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0f * ((float)i + 0.8f),
                    ((float4c*)(mem_cpu.data))[i].c.x, 1e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0f * ((float)i + 1.0f),
                    ((float4c*)(mem_cpu.data))[i].c.y, 1e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0f * ((float)i + 1.2f),
                    ((float4c*)(mem_cpu.data))[i].d.x, 1e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0f * ((float)i + 1.4f),
                    ((float4c*)(mem_cpu.data))[i].d.y, 1e-6);
        }

        // Copy to GPU.
        oskar_Mem mem_gpu(&mem_cpu, OSKAR_LOCATION_GPU);

        // Scale again.
        oskar_mem_scale_real(&mem_gpu, 2.0, &status);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);

        // Copy back and check contents.
        oskar_Mem mem_cpu2(&mem_gpu, OSKAR_LOCATION_CPU);
        for (int i = 0; i < n; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0f * ((float)i),
                    ((float4c*)(mem_cpu2.data))[i].a.x, 1e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0f * ((float)i + 0.2f),
                    ((float4c*)(mem_cpu2.data))[i].a.y, 1e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0f * ((float)i + 0.4f),
                    ((float4c*)(mem_cpu2.data))[i].b.x, 1e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0f * ((float)i + 0.6f),
                    ((float4c*)(mem_cpu2.data))[i].b.y, 1e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0f * ((float)i + 0.8f),
                    ((float4c*)(mem_cpu2.data))[i].c.x, 1e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0f * ((float)i + 1.0f),
                    ((float4c*)(mem_cpu2.data))[i].c.y, 1e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0f * ((float)i + 1.2f),
                    ((float4c*)(mem_cpu2.data))[i].d.x, 1e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0f * ((float)i + 1.4f),
                    ((float4c*)(mem_cpu2.data))[i].d.y, 1e-6);
        }
    }

    // Double precision real.
    {
        oskar_Mem mem_cpu(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, n);

        // Fill memory.
        for (int i = 0; i < n; ++i)
        {
            ((double*)(mem_cpu.data))[i] = (double)i;
        }

        // Scale.
        oskar_mem_scale_real(&mem_cpu, 2.0, &status);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);

        // Check contents.
        for (int i = 0; i < n; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0 * i,
                    ((double*)(mem_cpu.data))[i], 1e-12);
        }

        // Copy to GPU.
        oskar_Mem mem_gpu(&mem_cpu, OSKAR_LOCATION_GPU);

        // Scale again.
        oskar_mem_scale_real(&mem_gpu, 2.0, &status);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);

        // Copy back and check contents.
        oskar_Mem mem_cpu2(&mem_gpu, OSKAR_LOCATION_CPU);
        for (int i = 0; i < n; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0 * i,
                    ((double*)(mem_cpu2.data))[i], 1e-6);
        }
    }

    // Double precision complex.
    {
        oskar_Mem mem_cpu(OSKAR_DOUBLE_COMPLEX, OSKAR_LOCATION_CPU, n);

        // Fill memory.
        for (int i = 0; i < n; ++i)
        {
            ((double2*)(mem_cpu.data))[i].x = (double)i;
            ((double2*)(mem_cpu.data))[i].y = (double)i + 0.2;
        }

        // Scale.
        oskar_mem_scale_real(&mem_cpu, 2.0, &status);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);

        // Check contents.
        for (int i = 0; i < n; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0 * ((double)i),
                    ((double2*)(mem_cpu.data))[i].x, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0 * ((double)i + 0.2),
                    ((double2*)(mem_cpu.data))[i].y, 1e-12);
        }

        // Copy to GPU.
        oskar_Mem mem_gpu(&mem_cpu, OSKAR_LOCATION_GPU);

        // Scale again.
        oskar_mem_scale_real(&mem_gpu, 2.0, &status);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);

        // Copy back and check contents.
        oskar_Mem mem_cpu2(&mem_gpu, OSKAR_LOCATION_CPU);
        for (int i = 0; i < n; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0 * ((double)i),
                    ((double2*)(mem_cpu2.data))[i].x, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0 * ((double)i + 0.2),
                    ((double2*)(mem_cpu2.data))[i].y, 1e-12);
        }
    }

    // Double precision complex matrix.
    {
        oskar_Mem mem_cpu(OSKAR_DOUBLE_COMPLEX_MATRIX, OSKAR_LOCATION_CPU, n);

        // Fill memory.
        for (int i = 0; i < n; ++i)
        {
            ((double4c*)(mem_cpu.data))[i].a.x = (double)i;
            ((double4c*)(mem_cpu.data))[i].a.y = (double)i + 0.2;
            ((double4c*)(mem_cpu.data))[i].b.x = (double)i + 0.4;
            ((double4c*)(mem_cpu.data))[i].b.y = (double)i + 0.6;
            ((double4c*)(mem_cpu.data))[i].c.x = (double)i + 0.8;
            ((double4c*)(mem_cpu.data))[i].c.y = (double)i + 1.0;
            ((double4c*)(mem_cpu.data))[i].d.x = (double)i + 1.2;
            ((double4c*)(mem_cpu.data))[i].d.y = (double)i + 1.4;
        }

        // Scale.
        oskar_mem_scale_real(&mem_cpu, 2.0, &status);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);

        // Check contents.
        for (int i = 0; i < n; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0 * ((double)i),
                    ((double4c*)(mem_cpu.data))[i].a.x, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0 * ((double)i + 0.2),
                    ((double4c*)(mem_cpu.data))[i].a.y, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0 * ((double)i + 0.4),
                    ((double4c*)(mem_cpu.data))[i].b.x, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0 * ((double)i + 0.6),
                    ((double4c*)(mem_cpu.data))[i].b.y, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0 * ((double)i + 0.8),
                    ((double4c*)(mem_cpu.data))[i].c.x, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0 * ((double)i + 1.0),
                    ((double4c*)(mem_cpu.data))[i].c.y, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0 * ((double)i + 1.2),
                    ((double4c*)(mem_cpu.data))[i].d.x, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0 * ((double)i + 1.4),
                    ((double4c*)(mem_cpu.data))[i].d.y, 1e-12);
        }

        // Copy to GPU.
        oskar_Mem mem_gpu(&mem_cpu, OSKAR_LOCATION_GPU);

        // Scale again.
        oskar_mem_scale_real(&mem_gpu, 2.0, &status);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);

        // Copy back and check contents.
        oskar_Mem mem_cpu2(&mem_gpu, OSKAR_LOCATION_CPU);
        for (int i = 0; i < n; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0 * ((double)i),
                    ((double4c*)(mem_cpu2.data))[i].a.x, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0 * ((double)i + 0.2),
                    ((double4c*)(mem_cpu2.data))[i].a.y, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0 * ((double)i + 0.4),
                    ((double4c*)(mem_cpu2.data))[i].b.x, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0 * ((double)i + 0.6),
                    ((double4c*)(mem_cpu2.data))[i].b.y, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0 * ((double)i + 0.8),
                    ((double4c*)(mem_cpu2.data))[i].c.x, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0 * ((double)i + 1.0),
                    ((double4c*)(mem_cpu2.data))[i].c.y, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0 * ((double)i + 1.2),
                    ((double4c*)(mem_cpu2.data))[i].d.x, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0 * ((double)i + 1.4),
                    ((double4c*)(mem_cpu2.data))[i].d.y, 1e-12);
        }
    }
}

void Test_Mem::test_set_value_real()
{
    int n = 100, err = 0;

    // Double precision real.
    {
        oskar_Mem mem(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, n);
        oskar_mem_set_value_real(&mem, 4.5, &err);
        CPPUNIT_ASSERT_EQUAL(0, err);

        for (int i = 0; i < n; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((double*)mem)[i], 4.5, 1e-10);
        }
    }

    // Double precision complex.
    {
        oskar_Mem mem(OSKAR_DOUBLE_COMPLEX, OSKAR_LOCATION_GPU, n);
        oskar_mem_set_value_real(&mem, 6.5, &err);
        CPPUNIT_ASSERT_EQUAL(0, err);

        oskar_Mem mem2(&mem, OSKAR_LOCATION_CPU);
        for (int i = 0; i < n; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((double2*)mem2)[i].x, 6.5, 1e-10);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((double2*)mem2)[i].y, 0.0, 1e-10);
        }
    }

    // Double precision complex matrix.
    {
        oskar_Mem mem(OSKAR_DOUBLE_COMPLEX_MATRIX, OSKAR_LOCATION_GPU, n);
        oskar_mem_set_value_real(&mem, 6.5, &err);
        CPPUNIT_ASSERT_EQUAL(0, err);

        oskar_Mem mem2(&mem, OSKAR_LOCATION_CPU);
        for (int i = 0; i < n; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((double4c*)mem2)[i].a.x, 6.5, 1e-10);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((double4c*)mem2)[i].a.y, 0.0, 1e-10);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((double4c*)mem2)[i].b.x, 0.0, 1e-10);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((double4c*)mem2)[i].b.y, 0.0, 1e-10);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((double4c*)mem2)[i].c.x, 0.0, 1e-10);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((double4c*)mem2)[i].c.y, 0.0, 1e-10);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((double4c*)mem2)[i].d.x, 6.5, 1e-10);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((double4c*)mem2)[i].d.y, 0.0, 1e-10);
        }
    }

    // Single precision real.
    {
        oskar_Mem mem(OSKAR_SINGLE, OSKAR_LOCATION_CPU, n);
        oskar_mem_set_value_real(&mem, 4.5, &err);
        CPPUNIT_ASSERT_EQUAL(0, err);

        for (int i = 0; i < n; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((float*)mem)[i], 4.5, 1e-5);
        }
    }

    // Single precision complex.
    {
        oskar_Mem mem(OSKAR_SINGLE_COMPLEX, OSKAR_LOCATION_GPU, n);
        oskar_mem_set_value_real(&mem, 6.5, &err);
        CPPUNIT_ASSERT_EQUAL(0, err);

        oskar_Mem mem2(&mem, OSKAR_LOCATION_CPU);
        for (int i = 0; i < n; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((float2*)mem2)[i].x, 6.5, 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((float2*)mem2)[i].y, 0.0, 1e-5);
        }
    }

    // Single precision complex matrix.
    {
        oskar_Mem mem(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_LOCATION_GPU, n);
        oskar_mem_set_value_real(&mem, 6.5, &err);
        CPPUNIT_ASSERT_EQUAL(0, err);

        oskar_Mem mem2(&mem, OSKAR_LOCATION_CPU);
        for (int i = 0; i < n; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((float4c*)mem2)[i].a.x, 6.5, 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((float4c*)mem2)[i].a.y, 0.0, 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((float4c*)mem2)[i].b.x, 0.0, 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((float4c*)mem2)[i].b.y, 0.0, 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((float4c*)mem2)[i].c.x, 0.0, 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((float4c*)mem2)[i].c.y, 0.0, 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((float4c*)mem2)[i].d.x, 6.5, 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(((float4c*)mem2)[i].d.y, 0.0, 1e-5);
        }
    }
}

void Test_Mem::test_add()
{
    int error = 0;
    // Use case: Two CPU oskar_Mem matrix pointers are added together.
    {
        int num_elements = 10;
        oskar_Mem mem_A(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_LOCATION_CPU, num_elements);
        oskar_Mem mem_B(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_LOCATION_CPU, num_elements);
        float4c* A = (float4c*)mem_A.data;
        float4c* B = (float4c*)mem_B.data;

        for (int i = 0; i < num_elements; ++i)
        {
            A[i].a = make_float2((float)i + 0.1, (float)i + 0.2);
            A[i].b = make_float2((float)i + 0.3, (float)i + 0.4);
            A[i].c = make_float2((float)i + 0.5, (float)i + 0.6);
            A[i].d = make_float2((float)i + 0.7, (float)i + 0.8);
            B[i].a = make_float2(1.15, 0.15);
            B[i].b = make_float2(2.16, 0.16);
            B[i].c = make_float2(3.17, 0.17);
            B[i].d = make_float2(4.18, 0.18);
        }

        oskar_Mem mem_C(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_LOCATION_CPU, num_elements);
        oskar_mem_add(&mem_C, &mem_A, &mem_B, &error);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);

        float4c* C = (float4c*)mem_C.data;
        double delta = 1.0e-5;
        for (int i = 0; i < num_elements; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(A[i].a.x + B[i].a.x , C[i].a.x, delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(A[i].a.y + B[i].a.y , C[i].a.y, delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(A[i].b.x + B[i].b.x , C[i].b.x, delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(A[i].b.y + B[i].b.y , C[i].b.y, delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(A[i].c.x + B[i].c.x , C[i].c.x, delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(A[i].c.y + B[i].c.y , C[i].c.y, delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(A[i].d.x + B[i].d.x , C[i].d.x, delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(A[i].d.y + B[i].d.y , C[i].d.y, delta);
        }
    }

    // Use case: In place add.
    {
        double delta = 1.0e-5;

        int num_elements = 10;
        oskar_Mem mem_A(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_LOCATION_CPU, num_elements);
        oskar_Mem mem_B(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_LOCATION_CPU, num_elements);
        float4c* A = (float4c*)mem_A.data;
        float4c* B = (float4c*)mem_B.data;

        for (int i = 0; i < num_elements; ++i)
        {
            A[i].a = make_float2((float)i + 0.1, (float)i + 0.2);
            A[i].b = make_float2((float)i + 0.3, (float)i + 0.4);
            A[i].c = make_float2((float)i + 0.5, (float)i + 0.6);
            A[i].d = make_float2((float)i + 0.7, (float)i + 0.8);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, B[i].a.x, delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, B[i].a.y, delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, B[i].b.x, delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, B[i].b.y, delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, B[i].c.x, delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, B[i].c.y, delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, B[i].d.x, delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, B[i].d.y, delta);
        }

        oskar_mem_add(&mem_B, &mem_A, &mem_B, &error);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(error), 0, error);

        for (int i = 0; i < num_elements; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(A[i].a.x, B[i].a.x, delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(A[i].a.y, B[i].a.y, delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(A[i].b.x, B[i].b.x, delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(A[i].b.y, B[i].b.y, delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(A[i].c.x, B[i].c.x, delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(A[i].c.y, B[i].c.y, delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(A[i].d.x, B[i].d.x, delta);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(A[i].d.y, B[i].d.y, delta);
        }
    }

    // Use Case: memory on the GPU.
    {
        int num_elements = 10;
        oskar_Mem mem_A(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_LOCATION_GPU, num_elements);
        oskar_Mem mem_B(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_LOCATION_GPU, num_elements);
        oskar_Mem mem_C(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_LOCATION_GPU, num_elements);
        oskar_mem_add(&mem_C, &mem_A, &mem_B, &error);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_ERR_BAD_LOCATION, error);
        error = 0;
    }

    // Use Case: Dimension mismatch in mem pointers being added.
    {
        int num_elements = 10;
        oskar_Mem mem_A(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_LOCATION_CPU, num_elements);
        oskar_Mem mem_B(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_LOCATION_CPU, num_elements);
        oskar_Mem mem_C(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_LOCATION_CPU, 5);
        oskar_mem_add(&mem_C, &mem_A, &mem_B, &error);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_ERR_DIMENSION_MISMATCH, error);
        error = 0;
    }
}

void Test_Mem::test_add_noise()
{
    int num_elements = 1000;
    double stddev = 0.1;
    double mean = 5.0;
    int status = 0;

    // Test case: add Gaussian noise.
    {
        oskar_Mem values(OSKAR_DOUBLE_COMPLEX_MATRIX, OSKAR_LOCATION_CPU, num_elements);
        oskar_mem_add_gaussian_noise(&values, stddev, mean, &status);
        FILE* file;
        file = fopen("temp_mem_noise.dat", "wb");
        fwrite(values.data, sizeof(double4c), num_elements, file);
        fclose(file);
    }
}

void Test_Mem::test_binary()
{
    // Remove the file if it already exists.
    const char filename[] = "cpp_unit_test_mem_binary.dat";
    if (oskar_file_exists(filename))
        remove(filename);
    int num_cpu = 1000;
    int num_gpu = 2048;
    int error = 0;

    // Save data from CPU.
    {
        oskar_Mem mem_cpu(OSKAR_SINGLE, OSKAR_LOCATION_CPU, num_cpu);
        float* data = (float*)mem_cpu;

        // Fill array with data.
        for (int i = 0; i < num_cpu; ++i)
        {
            data[i] = i * 1024.0;
        }

        // Save CPU data.
        error = oskar_mem_binary_file_write_ext(&mem_cpu, filename,
                "USER", "TEST", 987654, 0);
        CPPUNIT_ASSERT_EQUAL(0, error);
    }

    // Save data from GPU.
    {
        oskar_Mem mem_cpu(OSKAR_DOUBLE_COMPLEX, OSKAR_LOCATION_CPU, num_gpu);
        oskar_Mem mem_gpu(OSKAR_DOUBLE_COMPLEX, OSKAR_LOCATION_GPU, num_gpu);
        double2* data = (double2*)mem_cpu;

        // Fill array with data.
        for (int i = 0; i < num_gpu; ++i)
        {
            data[i].x = i * 10.0;
            data[i].y = i * 20.0 + 1.0;
        }

        // Copy data to GPU.
        oskar_mem_copy(&mem_gpu, &mem_cpu, &error);
        CPPUNIT_ASSERT_EQUAL(0, error);

        // Save GPU data.
        error = oskar_mem_binary_file_write_ext(&mem_gpu, filename,
                "AA", "BB", 2, 0);
        CPPUNIT_ASSERT_EQUAL(0, error);
    }

    // Save a single integer with a large index.
    int val = 0xFFFFFF;
    error = oskar_binary_file_write_int(filename, 50, 9, 800000, val);
    CPPUNIT_ASSERT_EQUAL(0, error);

    // Save data from CPU with blank tags.
    {
        oskar_Mem mem_cpu(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, 2 * num_cpu);
        double* data = (double*)mem_cpu;

        // Fill array with data.
        for (int i = 0; i < 2 * num_cpu; ++i)
        {
            data[i] = i * 500.0;
        }

        // Save CPU data.
        error = oskar_mem_binary_file_write_ext(&mem_cpu, filename,
                "", "", 10, 0);
        CPPUNIT_ASSERT_EQUAL(0, error);

        // Fill array with data.
        for (int i = 0; i < 2 * num_cpu; ++i)
        {
            data[i] = i * 501.0;
        }

        // Save CPU data.
        error = oskar_mem_binary_file_write_ext(&mem_cpu, filename,
                "", "", 11, 0);
        CPPUNIT_ASSERT_EQUAL(0, error);
    }

    // Save CPU data with tags that are equal lengths.
    {
        oskar_Mem mem_cpu(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_cpu);
        double* data = (double*)mem_cpu;

        // Fill array with data.
        for (int i = 0; i < num_cpu; ++i)
        {
            data[i] = i * 1001.0;
        }

        // Save CPU data.
        error = oskar_mem_binary_file_write_ext(&mem_cpu, filename,
                "DOG", "CAT", 0, 0);
        CPPUNIT_ASSERT_EQUAL(0, error);

        // Fill array with data.
        for (int i = 0; i < num_cpu; ++i)
        {
            data[i] = i * 127.0;
        }

        // Save CPU data.
        error = oskar_mem_binary_file_write_ext(&mem_cpu, filename,
                "ONE", "TWO", 0, 0);
        CPPUNIT_ASSERT_EQUAL(0, error);
    }

    // Declare index pointer.
    oskar_BinaryTagIndex* index = NULL;

    // Load GPU data.
    {
        oskar_Mem mem_gpu(OSKAR_DOUBLE_COMPLEX, OSKAR_LOCATION_GPU);
        error = oskar_mem_binary_file_read_ext(&mem_gpu, filename, &index,
                "AA", "BB", 2);
        CPPUNIT_ASSERT_EQUAL(0, error);
        CPPUNIT_ASSERT_EQUAL(num_gpu, mem_gpu.num_elements);

        // Copy back to CPU and examine contents.
        oskar_Mem mem_cpu(&mem_gpu, OSKAR_LOCATION_CPU);
        double2* data = (double2*)mem_cpu;
        for (int i = 0; i < num_gpu; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(i * 10.0,       data[i].x, 1e-8);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(i * 20.0 + 1.0, data[i].y, 1e-8);
        }
    }

    // Load integer with a large index.
    int new_val = 0;
    error = oskar_binary_file_read_int(filename, &index,
            50, 9, 800000, &new_val);
    CPPUNIT_ASSERT_EQUAL(0, error);
    CPPUNIT_ASSERT_EQUAL(val, new_val);

    // Load CPU data.
    {
        oskar_Mem mem_cpu(OSKAR_SINGLE, OSKAR_LOCATION_CPU, num_cpu);
        error = oskar_mem_binary_file_read_ext(&mem_cpu, filename, &index,
                "USER", "TEST", 987654);
        CPPUNIT_ASSERT_EQUAL(0, error);
        CPPUNIT_ASSERT_EQUAL(num_cpu, mem_cpu.num_elements);
        float* data = (float*)mem_cpu;
        for (int i = 0; i < num_cpu; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(i * 1024.0, data[i], 1e-8);
        }
    }

    // Load CPU data with blank tags.
    {
        double* data;
        oskar_Mem mem_cpu(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, 2 * num_cpu);
        error = oskar_mem_binary_file_read_ext(&mem_cpu, filename, &index,
                "", "", 10);
        CPPUNIT_ASSERT_EQUAL(0, error);
        error = oskar_mem_binary_file_read_ext(&mem_cpu, filename, &index,
                "DOESN'T", "EXIST", 10);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_ERR_BINARY_TAG_NOT_FOUND, error);
        CPPUNIT_ASSERT_EQUAL(2 * num_cpu, mem_cpu.num_elements);
        data = (double*)mem_cpu;
        for (int i = 0; i < 2 * num_cpu; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(i * 500.0, data[i], 1e-8);
        }
        error = oskar_mem_binary_file_read_ext(&mem_cpu, filename, &index,
                "", "", 11);
        CPPUNIT_ASSERT_EQUAL(0, error);
        CPPUNIT_ASSERT_EQUAL(2 * num_cpu, mem_cpu.num_elements);
        data = (double*)mem_cpu;
        for (int i = 0; i < 2 * num_cpu; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(i * 501.0, data[i], 1e-8);
        }
    }

    // Load CPU data with tags that are equal lengths.
    {
        double* data;
        oskar_Mem mem_cpu(OSKAR_DOUBLE, OSKAR_LOCATION_CPU);
        error = oskar_mem_binary_file_read_ext(&mem_cpu, filename, &index,
                "ONE", "TWO", 0);
        CPPUNIT_ASSERT_EQUAL(0, error);
        CPPUNIT_ASSERT_EQUAL(num_cpu, mem_cpu.num_elements);
        data = (double*)mem_cpu;
        for (int i = 0; i < num_cpu; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(i * 127.0, data[i], 1e-8);
        }
        error = oskar_mem_binary_file_read_ext(&mem_cpu, filename, &index,
                "DOG", "CAT", 0);
        CPPUNIT_ASSERT_EQUAL(0, error);
        CPPUNIT_ASSERT_EQUAL(num_cpu, mem_cpu.num_elements);
        data = (double*)mem_cpu;
        for (int i = 0; i < num_cpu; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(i * 1001.0, data[i], 1e-8);
        }
    }

    // Try to load data that isn't present.
    {
        oskar_Mem mem_cpu(OSKAR_DOUBLE, OSKAR_LOCATION_CPU);
        error = oskar_mem_binary_file_read_ext(&mem_cpu, filename, &index,
                "DOESN'T", "EXIST", 10);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_ERR_BINARY_TAG_NOT_FOUND, error);
        CPPUNIT_ASSERT_EQUAL(0, mem_cpu.num_elements);
    }

    // Free the tag index.
    oskar_binary_tag_index_free(&index);
}

void Test_Mem::test_copy_gpu()
{
    int n = 100;
    oskar_Mem cpu(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, n, OSKAR_TRUE);
    double* cpu_ = (double*)cpu.data;
    for (int i = 0; i < n; ++i)
    {
        cpu_[i] = (double)i;
    }

    {
        oskar_Mem gpu(&cpu, OSKAR_LOCATION_GPU);
        oskar_Mem cpu2(&gpu, OSKAR_LOCATION_CPU);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(cpu_[5],((double*)cpu2.data)[5], 1e-6);
    }

    cudaDeviceReset();
}

