/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#include <oskar_BinaryTag.h>
#include <oskar_binary_file_read.h>
#include <oskar_binary_file_write.h>
#include <oskar_binary_stream_read_header.h>
#include <oskar_binary_stream_read.h>
#include <oskar_binary_stream_write_header.h>
#include <oskar_binary_stream_write.h>
#include <oskar_binary_tag_index_create.h>
#include <oskar_binary_tag_index_free.h>
#include <oskar_binary_header_version.h>
#include <oskar_endian.h>
#include <oskar_file_exists.h>
#include <oskar_get_error_string.h>
#include <oskar_mem.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

TEST(binary_file, test_stream)
{
    char filename[] = "temp_test_binary_stream.dat";
    int status = 0;

    // Create some data.
    int a1 = 65, b1 = 66, c1 = 67;
    int a = 0, b = 0, c = 0;
    int num_elements_double = 22;
    int num_elements_int = 17;
    size_t size_double = num_elements_double * sizeof(double);
    size_t size_int = num_elements_int * sizeof(int);

    // Write the file.
    {
        // Create some test data.
        std::vector<double> data_double(num_elements_double);
        std::vector<int> data_int(num_elements_int);

        // Open a file for binary write.
        FILE* stream = fopen(filename, "wb");

        // Write the header.
        oskar_binary_stream_write_header(stream, &status);

        // Write data.
        oskar_binary_stream_write_int(stream, 0, 0, 12345, a1, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        {
            for (int i = 0; i < num_elements_double; ++i)
                data_double[i] = i + 1000.0;
            oskar_binary_stream_write(stream, OSKAR_DOUBLE,
                    1, 10, 987654321, size_double, &data_double[0], &status);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
        }
        {
            for (int i = 0; i < num_elements_int; ++i)
                data_int[i] = i * 10;
            oskar_binary_stream_write(stream, OSKAR_INT,
                    2, 20, 1, size_int, &data_int[0], &status);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
        }
        oskar_binary_stream_write_int(stream, 0, 0, 2, c1, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        {
            for (int i = 0; i < num_elements_int; ++i)
                data_int[i] = i * 75;
            oskar_binary_stream_write(stream, OSKAR_INT,
                    14, 5, 6, size_int, &data_int[0], &status);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
        }
        oskar_binary_stream_write_int(stream, 12, 0, 0, b1, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        {
            for (int i = 0; i < num_elements_double; ++i)
                data_double[i] = i * 1234.0;
            oskar_binary_stream_write(stream, OSKAR_DOUBLE,
                    4, 0, 3, size_double, &data_double[0], &status);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
        }

        // Close the file.
        fclose(stream);
    }

    // Read the file back again.
    FILE* stream = fopen(filename, "rb");

    // Read the header back.
    oskar_BinaryHeader header;
    oskar_binary_stream_read_header(stream, &header, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Check the contents of the header.
    ASSERT_EQ(0, strcmp("OSKARBIN", header.magic));
    ASSERT_EQ(header.bin_version, (char)OSKAR_BINARY_FORMAT_VERSION);
    ASSERT_EQ(header.endian, (char)oskar_endian());
    ASSERT_EQ(header.size_ptr, (char)sizeof(void*));
    ASSERT_EQ(header.size_int, (char)sizeof(int));
    ASSERT_EQ(header.size_long, (char)sizeof(long));
    ASSERT_EQ(header.size_float, (char)sizeof(float));
    ASSERT_EQ(header.size_double, (char)sizeof(double));
    ASSERT_EQ(oskar_binary_header_version(&header),
            (int)OSKAR_VERSION);

    // Create the tag index.
    oskar_BinaryTagIndex* idx = NULL;
    oskar_binary_tag_index_create(&idx, stream, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    EXPECT_EQ(7, idx->num_tags);

    // Read the single numbers back and check values.
    oskar_binary_stream_read_int(stream, &idx, 0, 0, 12345, &a, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    EXPECT_EQ(a1, a);
    oskar_binary_stream_read_int(stream, &idx, 12, 0, 0, &b, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    EXPECT_EQ(b1, b);
    oskar_binary_stream_read_int(stream, &idx, 0, 0, 2, &c, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    EXPECT_EQ(c1, c);

    // Read the arrays back and check values.
    {
        std::vector<double> data_double(num_elements_double);
        std::vector<int> data_int(num_elements_int);
        oskar_binary_stream_read(stream, &idx, OSKAR_INT,
                2, 20, 1, size_int, &data_int[0], &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_binary_stream_read(stream, &idx, OSKAR_DOUBLE,
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
        oskar_binary_stream_read(stream, &idx, OSKAR_INT,
                14, 5, 6, size_int, &data_int[0], &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_binary_stream_read(stream, &idx, OSKAR_DOUBLE,
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
        oskar_binary_stream_read_double(stream, &idx, 255, 0, 0, &t, &status);
        EXPECT_EQ((int)OSKAR_ERR_BINARY_TAG_NOT_FOUND, status);
        status = 0;
    }

    // Close the file.
    fclose(stream);

    // Free the tag index.
    oskar_binary_tag_index_free(idx, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Remove the file.
    remove(filename);
}

TEST(binary_file, test_file)
{
    char filename[] = "temp_test_binary_file.dat";
    int status = 0;

    // If the file exists, delete it.
    if (oskar_file_exists(filename))
        remove(filename);

    // Create some data.
    int a1 = 65, b1 = 66, c1 = 67;
    int a = 0, b = 0, c = 0;
    int num_elements_double = 22;
    int num_elements_int = 17;
    size_t size_double = num_elements_double * sizeof(double);
    size_t size_int = num_elements_int * sizeof(int);

    // Write the file.
    {
        // Create some test data.
        std::vector<double> data_double(num_elements_double);
        std::vector<int> data_int(num_elements_int);

        // Write data.
        oskar_binary_file_write_int(filename, 0, 0, 12345, a1, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        {
            for (int i = 0; i < num_elements_double; ++i)
                data_double[i] = i + 1000.0;
            oskar_binary_file_write(filename, OSKAR_DOUBLE,
                    1, 10, 987654321, size_double, &data_double[0], &status);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
        }
        {
            for (int i = 0; i < num_elements_int; ++i)
                data_int[i] = i * 10;
            oskar_binary_file_write(filename, OSKAR_INT,
                    2, 20, 1, size_int, &data_int[0], &status);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
        }
        oskar_binary_file_write_int(filename, 0, 0, 2, c1, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        {
            for (int i = 0; i < num_elements_int; ++i)
                data_int[i] = i * 75;
            oskar_binary_file_write(filename, OSKAR_INT,
                    14, 5, 6, size_int, &data_int[0], &status);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
        }
        oskar_binary_file_write_int(filename, 12, 0, 0, b1, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        {
            for (int i = 0; i < num_elements_double; ++i)
                data_double[i] = i * 1234.0;
            oskar_binary_file_write(filename, OSKAR_DOUBLE,
                    4, 0, 3, size_double, &data_double[0], &status);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
        }
    }

    // Read the single numbers back and check values.
    oskar_BinaryTagIndex* idx = NULL;
    oskar_binary_file_read_int(filename, &idx, 0, 0, 12345, &a, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    EXPECT_EQ(a1, a);
    oskar_binary_file_read_int(filename, &idx, 12, 0, 0, &b, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    EXPECT_EQ(b1, b);
    oskar_binary_file_read_int(filename, &idx, 0, 0, 2, &c, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    EXPECT_EQ(c1, c);

    // Read the arrays back and check values.
    {
        std::vector<double> data_double(num_elements_double);
        std::vector<int> data_int(num_elements_int);
        oskar_binary_file_read(filename, &idx, OSKAR_INT,
                2, 20, 1, size_int, &data_int[0], &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_binary_file_read(filename, &idx, OSKAR_DOUBLE,
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
        oskar_binary_file_read(filename, &idx, OSKAR_INT,
                14, 5, 6, size_int, &data_int[0], &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_binary_file_read(filename, &idx, OSKAR_DOUBLE,
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
        oskar_binary_file_read_double(filename, &idx, 255, 0, 0, &t, &status);
        EXPECT_EQ((int)OSKAR_ERR_BINARY_TAG_NOT_FOUND, status);
        status = 0;
    }

    // Free the tag index.
    oskar_binary_tag_index_free(idx, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Remove the file.
    remove(filename);
}
