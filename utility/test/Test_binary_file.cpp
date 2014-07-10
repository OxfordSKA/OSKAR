/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#include <gtest/gtest.h>

#include <private_binary.h>
#include <oskar_binary.h>
#include <oskar_endian.h>
#include <oskar_file_exists.h>
#include <oskar_get_error_string.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

TEST(binary_file, test_file)
{
    char filename[] = "temp_test_binary_file.dat";
    int status = 0;

    // Create some data.
    int a1 = 65, b1 = 66, c1 = 67;
    int a = 0, b = 0, c = 0;
    int num_elements_double = 22;
    int num_elements_int = 17;
    size_t size_double = num_elements_double * sizeof(double);
    size_t size_int = num_elements_int * sizeof(int);

    // Create the handle.
    oskar_Binary* h = oskar_binary_create(filename, 'w', &status);

    // Write the file.
    {
        // Create some test data.
        std::vector<double> data_double(num_elements_double);
        std::vector<int> data_int(num_elements_int);

        // Write data.
        oskar_binary_write_int(h, 0, 0, 12345, a1, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        {
            for (int i = 0; i < num_elements_double; ++i)
                data_double[i] = i + 1000.0;
            oskar_binary_write(h, OSKAR_DOUBLE,
                    1, 10, 987654321, size_double, &data_double[0], &status);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
        }
        {
            for (int i = 0; i < num_elements_int; ++i)
                data_int[i] = i * 10;
            oskar_binary_write(h, OSKAR_INT,
                    2, 20, 1, size_int, &data_int[0], &status);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
        }
        oskar_binary_write_int(h, 0, 0, 2, c1, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        {
            for (int i = 0; i < num_elements_int; ++i)
                data_int[i] = i * 75;
            oskar_binary_write(h, OSKAR_INT,
                    14, 5, 6, size_int, &data_int[0], &status);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
        }
        oskar_binary_write_int(h, 12, 0, 0, b1, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        {
            for (int i = 0; i < num_elements_double; ++i)
                data_double[i] = i * 1234.0;
            oskar_binary_write(h, OSKAR_DOUBLE,
                    4, 0, 3, size_double, &data_double[0], &status);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
        }
    }

    oskar_binary_free(h);

    // Read the single numbers back and check values.
    h = oskar_binary_create(filename, 'r', &status);
    oskar_binary_read_int(h, 0, 0, 12345, &a, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    EXPECT_EQ(a1, a);
    oskar_binary_read_int(h, 12, 0, 0, &b, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    EXPECT_EQ(b1, b);
    oskar_binary_read_int(h, 0, 0, 2, &c, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    EXPECT_EQ(c1, c);

    // Read the arrays back and check values.
    {
        std::vector<double> data_double(num_elements_double);
        std::vector<int> data_int(num_elements_int);
        oskar_binary_read(h, OSKAR_INT,
                2, 20, 1, size_int, &data_int[0], &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_binary_read(h, OSKAR_DOUBLE,
                4, 0, 3, size_double, &data_double[0], &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        for (int i = 0; i < num_elements_double; ++i)
            EXPECT_DOUBLE_EQ(i * 1234.0, data_double[i]);
        for (int i = 0; i < num_elements_int; ++i)
            EXPECT_EQ(i * 10, data_int[i]);
    }
    {
        std::vector<double> data_double(num_elements_double);
        std::vector<int> data_int(num_elements_int);
        oskar_binary_read(h, OSKAR_INT,
                14, 5, 6, size_int, &data_int[0], &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_binary_read(h, OSKAR_DOUBLE,
                1, 10, 987654321, size_double, &data_double[0], &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        for (int i = 0; i < num_elements_double; ++i)
            EXPECT_DOUBLE_EQ(i + 1000.0, data_double[i]);
        for (int i = 0; i < num_elements_int; ++i)
            EXPECT_EQ(i * 75, data_int[i]);
    }

    // Look for a tag that isn't there.
    {
        double t;
        oskar_binary_read_double(h, 255, 0, 0, &t, &status);
        EXPECT_EQ((int)OSKAR_ERR_BINARY_TAG_NOT_FOUND, status);
        status = 0;
    }

    // Free the handle.
    oskar_binary_free(h);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Remove the file.
    remove(filename);
}

TEST(binary_file, binary_read_write_mem)
{
    const char filename[] = "temp_test_mem_binary.dat";
    int num_cpu = 1000;
    int num_gpu = 2048;
    int status = 0;

    // Create the handle.
    oskar_Binary* h = oskar_binary_create(filename, 'w', &status);

    // Save data from CPU.
    {
        oskar_Mem* mem = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU,
                num_cpu, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        float* data = oskar_mem_float(mem, &status);

        // Fill array with data.
        for (int i = 0; i < num_cpu; ++i)
        {
            data[i] = i * 1024.0;
        }

        // Save CPU data.
        oskar_binary_write_mem_ext(h, mem, "USER", "TEST", 987654, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_mem_free(mem, &status);
    }

    // Save data from GPU.
    {
        oskar_Mem *mem_cpu, *mem_gpu;
        mem_cpu = oskar_mem_create(OSKAR_DOUBLE_COMPLEX, OSKAR_CPU,
                num_gpu, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        double2* data = oskar_mem_double2(mem_cpu, &status);

        // Fill array with data.
        for (int i = 0; i < num_gpu; ++i)
        {
            data[i].x = i * 10.0;
            data[i].y = i * 20.0 + 1.0;
        }

        // Copy data to GPU.
        mem_gpu = oskar_mem_create_copy(mem_cpu, OSKAR_GPU, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Save GPU data.
        oskar_binary_write_mem_ext(h, mem_gpu, "AA", "BB", 2, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_mem_free(mem_cpu, &status);
        oskar_mem_free(mem_gpu, &status);
    }

    // Save a single integer with a large index.
    int val = 0xFFFFFF;
    oskar_binary_write_int(h, 50, 9, 800000, val, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Save data from CPU with blank tags.
    {
        oskar_Mem* mem = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU,
                num_cpu, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        double* data = oskar_mem_double(mem, &status);

        // Fill array with data.
        for (int i = 0; i < num_cpu; ++i)
        {
            data[i] = i * 500.0;
        }

        // Save CPU data.
        oskar_binary_write_mem_ext(h, mem, "", "", 10, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Fill array with data.
        for (int i = 0; i < num_cpu; ++i)
        {
            data[i] = i * 501.0;
        }

        // Save CPU data.
        oskar_binary_write_mem_ext(h, mem, "", "", 11, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_mem_free(mem, &status);
    }

    // Save CPU data with tags that are equal lengths.
    {
        oskar_Mem* mem = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU,
                num_cpu, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        double* data = oskar_mem_double(mem, &status);

        // Fill array with data.
        for (int i = 0; i < num_cpu; ++i)
        {
            data[i] = i * 1001.0;
        }

        // Save CPU data.
        oskar_binary_write_mem_ext(h, mem, "DOG", "CAT", 0, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Fill array with data.
        for (int i = 0; i < num_cpu; ++i)
        {
            data[i] = i * 127.0;
        }

        // Save CPU data.
        oskar_binary_write_mem_ext(h, mem, "ONE", "TWO", 0, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_mem_free(mem, &status);
    }

    // Create the handle for reading.
    oskar_binary_free(h);
    h = oskar_binary_create(filename, 'r', &status);

    // Load data directly to GPU.
    {
        oskar_Mem *mem_gpu, *mem_cpu;
        mem_gpu = oskar_mem_create(OSKAR_DOUBLE_COMPLEX, OSKAR_GPU,
                0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_binary_read_mem_ext(h, mem_gpu, "AA", "BB", 2, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        EXPECT_EQ(num_gpu, (int)oskar_mem_length(mem_gpu));

        // Copy back to CPU and examine contents.
        mem_cpu = oskar_mem_create_copy(mem_gpu, OSKAR_CPU, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        double2* data = oskar_mem_double2(mem_cpu, &status);
        for (int i = 0; i < num_gpu; ++i)
        {
            EXPECT_DOUBLE_EQ(i * 10.0,       data[i].x);
            EXPECT_DOUBLE_EQ(i * 20.0 + 1.0, data[i].y);
        }
        oskar_mem_free(mem_cpu, &status);
        oskar_mem_free(mem_gpu, &status);
    }

    // Load integer with a large index.
    int new_val = 0;
    oskar_binary_read_int(h, 50, 9, 800000, &new_val, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    EXPECT_EQ(val, new_val);

    // Load CPU data.
    {
        oskar_Mem* mem = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU,
                num_cpu, &status);
        oskar_binary_read_mem_ext(h, mem, "USER", "TEST", 987654, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(num_cpu, (int)oskar_mem_length(mem));
        float* data = oskar_mem_float(mem, &status);
        for (int i = 0; i < num_cpu; ++i)
        {
            EXPECT_DOUBLE_EQ(i * 1024.0, data[i]);
        }
        oskar_mem_free(mem, &status);
    }

    // Load CPU data with blank tags.
    {
        double* data;
        oskar_Mem* mem = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU,
                num_cpu, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_binary_read_mem_ext(h, mem, "", "", 10, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_binary_read_mem_ext(h, mem, "DOESN'T", "EXIST", 10, &status);
        EXPECT_EQ((int)OSKAR_ERR_BINARY_TAG_NOT_FOUND, status);
        status = 0;
        ASSERT_EQ(num_cpu, (int)oskar_mem_length(mem));
        data = oskar_mem_double(mem, &status);
        for (int i = 0; i < num_cpu; ++i)
        {
            EXPECT_DOUBLE_EQ(i * 500.0, data[i]);
        }
        oskar_binary_read_mem_ext(h, mem, "", "", 11, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(num_cpu, (int)oskar_mem_length(mem));
        data = oskar_mem_double(mem, &status);
        for (int i = 0; i < num_cpu; ++i)
        {
            EXPECT_DOUBLE_EQ(i * 501.0, data[i]);
        }
        oskar_mem_free(mem, &status);
    }

    // Load CPU data with tags that are equal lengths.
    {
        double* data;
        oskar_Mem* mem = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0,
                &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_binary_read_mem_ext(h, mem, "ONE", "TWO", 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(num_cpu, (int)oskar_mem_length(mem));
        data = oskar_mem_double(mem, &status);
        for (int i = 0; i < num_cpu; ++i)
        {
            EXPECT_DOUBLE_EQ(i * 127.0, data[i]);
        }
        oskar_binary_read_mem_ext(h, mem, "DOG", "CAT", 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(num_cpu, (int)oskar_mem_length(mem));
        data = oskar_mem_double(mem, &status);
        for (int i = 0; i < num_cpu; ++i)
        {
            EXPECT_DOUBLE_EQ(i * 1001.0, data[i]);
        }
        oskar_mem_free(mem, &status);
    }

    // Try to load data that isn't present.
    {
        oskar_Mem* mem = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0,
                &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_binary_read_mem_ext(h, mem, "DOESN'T", "EXIST", 10, &status);
        EXPECT_EQ((int)OSKAR_ERR_BINARY_TAG_NOT_FOUND, status);
        status = 0;
        EXPECT_EQ(0, (int)oskar_mem_length(mem));
        oskar_mem_free(mem, &status);
    }

    // Check header version.
    {
        int maj, min, patch;
        oskar_binary_read_oskar_version(filename, &maj, &min, &patch, &status);
        EXPECT_EQ(OSKAR_VERSION & 0xFF, patch);
        EXPECT_EQ((OSKAR_VERSION & 0xFF00) >> 8, min);
        EXPECT_EQ((OSKAR_VERSION & 0xFF0000) >> 16, maj);
    }

    // Release the handle.
    oskar_binary_free(h);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}

