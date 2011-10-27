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

#include "utility/test/oskar_Mem_test.h"
#include "utility/oskar_mem_realloc.h"
#include "utility/oskar_mem_append.h"
#include "utility/oskar_Mem.h"
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
using namespace std;

void oskar_Mem_test::test_alloc()
{
}

void oskar_Mem_test::test_realloc()
{
    int error = 0;

    oskar_Mem mem_gpu(OSKAR_DOUBLE, OSKAR_LOCATION_GPU, 0);
    error = oskar_mem_realloc(&mem_gpu, 500);
    CPPUNIT_ASSERT_EQUAL_MESSAGE((error > 0) ? std::string("CUDA ERROR: ") +
            cudaGetErrorString((cudaError_t)error) : "OSKAR ERROR", 0, error);
    CPPUNIT_ASSERT_EQUAL(500, mem_gpu.n_elements());
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_DOUBLE, mem_gpu.type());

    oskar_Mem mem_cpu(OSKAR_DOUBLE_COMPLEX, OSKAR_LOCATION_CPU, 100);
    error = oskar_mem_realloc(&mem_cpu, 1000);
    CPPUNIT_ASSERT_EQUAL_MESSAGE((error > 0) ? std::string("CUDA ERROR: ") +
            cudaGetErrorString((cudaError_t)error) : "OSKAR ERROR", 0, error);
    CPPUNIT_ASSERT_EQUAL(1000, mem_cpu.n_elements());
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_DOUBLE_COMPLEX, mem_cpu.type());
}


void oskar_Mem_test::test_append()
{
    {
        oskar_Mem mem_cpu(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, 0);
        int num_values1 = 10;
        double value1 = 1.0;
        vector<double> data1(num_values1, value1);
        mem_cpu.append((const void*)&data1[0], OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_values1);
        CPPUNIT_ASSERT_EQUAL(num_values1, mem_cpu.n_elements());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_CPU, mem_cpu.location());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_DOUBLE, mem_cpu.type());
        for (int i = 0; i < mem_cpu.n_elements(); ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(value1, ((double*)mem_cpu.data)[i], 1.0e-5);
        }
        int num_values2 = 5;
        double value2 = 2.0;
        vector<double> data2(num_values2, value2);
        mem_cpu.append((const void*)&data2[0], OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_values2);
        CPPUNIT_ASSERT_EQUAL(num_values1 + num_values2, mem_cpu.n_elements());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_CPU, mem_cpu.location());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_DOUBLE, mem_cpu.type());
        for (int i = 0; i < mem_cpu.n_elements(); ++i)
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
        int error = 0;
        error = mem_gpu.append((const void*)&data1[0], OSKAR_SINGLE, OSKAR_LOCATION_CPU, num_values1);
        CPPUNIT_ASSERT_EQUAL_MESSAGE((error > 0) ? std::string("CUDA ERROR: ") +
                cudaGetErrorString((cudaError_t)error) : "OSKAR ERROR", 0, error);
        CPPUNIT_ASSERT_EQUAL(num_values1, mem_gpu.n_elements());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_GPU, mem_gpu.location());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, mem_gpu.type());
        oskar_Mem mem_temp(&mem_gpu, OSKAR_LOCATION_CPU);
        for (int i = 0; i < mem_gpu.n_elements(); ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(value1, ((float*)mem_temp.data)[i], 1.0e-5);
        }

        int num_values2 = 5;
        float value2 = 2.0;
        vector<float> data2(num_values2, value2);
        error = mem_gpu.append((const void*)&data2[0], OSKAR_SINGLE, OSKAR_LOCATION_CPU, num_values2);
        CPPUNIT_ASSERT_EQUAL_MESSAGE((error > 0) ? std::string("CUDA ERROR: ") +
                cudaGetErrorString((cudaError_t)error) : "OSKAR ERROR", 0, error);
        CPPUNIT_ASSERT_EQUAL(num_values1 + num_values2, mem_gpu.n_elements());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_GPU, mem_gpu.location());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, mem_gpu.type());
        oskar_Mem mem_temp2(&mem_gpu, OSKAR_LOCATION_CPU);
        for (int i = 0; i < mem_gpu.n_elements(); ++i)
        {
            if (i < num_values1)
                CPPUNIT_ASSERT_DOUBLES_EQUAL(value1, ((float*)mem_temp2.data)[i], 1.0e-5);
            else
                CPPUNIT_ASSERT_DOUBLES_EQUAL(value2, ((float*)mem_temp2.data)[i], 1.0e-5);
        }
    }
}
