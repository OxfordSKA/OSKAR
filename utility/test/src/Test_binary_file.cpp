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

#include "utility/test/Test_binary_file.h"
#include "utility/oskar_BinaryTag.h"
#include "utility/oskar_binary_stream_read_header.h"
#include "utility/oskar_binary_stream_read.h"
#include "utility/oskar_binary_stream_write_header.h"
#include "utility/oskar_binary_stream_write.h"
#include "utility/oskar_binary_tag_index_create.h"
#include "utility/oskar_binary_tag_index_free.h"
#include "utility/oskar_binary_header_version.h"
#include "utility/oskar_endian.h"
#include "utility/oskar_Mem.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

void Test_binary_file::test_method()
{
    char filename[] = "cpp_unit_test_binary.dat";
    int err;

    // Create some data.
    int num_times = 65, num_channels = 66, num_baselines = 67;
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
        FILE* file = fopen(filename, "wb");

        // Write the header.
        oskar_binary_stream_write_header(file);

        // Write data.
        err = oskar_binary_stream_write_int(file, OSKAR_TAG_NUM_TIMES, 0, 0,
                num_times);
        CPPUNIT_ASSERT_EQUAL(0, err);
        {
            for (int i = 0; i < num_elements_double; ++i)
                data_double[i] = i + 1000.0;
            err = oskar_binary_stream_write(file, OSKAR_TAG_USER, 2, 0,
                    OSKAR_DOUBLE, size_double, &data_double[0]);
            CPPUNIT_ASSERT_EQUAL(0, err);
        }
        {
            for (int i = 0; i < num_elements_int; ++i)
                data_int[i] = i * 10;
            err = oskar_binary_stream_write(file, OSKAR_TAG_USER, 10, 0,
                    OSKAR_INT, size_int, &data_int[0]);
            CPPUNIT_ASSERT_EQUAL(0, err);
        }
        err = oskar_binary_stream_write_int(file, OSKAR_TAG_NUM_BASELINES, 0, 0,
                num_baselines);
        CPPUNIT_ASSERT_EQUAL(0, err);
        {
            for (int i = 0; i < num_elements_int; ++i)
                data_int[i] = i * 75;
            err = oskar_binary_stream_write(file, OSKAR_TAG_USER, 14, 5,
                    OSKAR_INT, size_int, &data_int[0]);
            CPPUNIT_ASSERT_EQUAL(0, err);
        }
        err = oskar_binary_stream_write_int(file, OSKAR_TAG_NUM_CHANNELS, 0, 0,
                num_channels);
        CPPUNIT_ASSERT_EQUAL(0, err);
        {
            for (int i = 0; i < num_elements_double; ++i)
                data_double[i] = i * 1234.0;
            err = oskar_binary_stream_write(file, OSKAR_TAG_USER, 4, 0,
                    OSKAR_DOUBLE, size_double, &data_double[0]);
        }

        // Close the file.
        fclose(file);
    }

    // Read the file back again.
    FILE* file = fopen(filename, "rb");

    // Read the header back.
    oskar_BinaryHeader header;
    err = oskar_binary_stream_read_header(file, &header);
    CPPUNIT_ASSERT_EQUAL(0, err);

    // Check the contents of the header.
    CPPUNIT_ASSERT_EQUAL(0, strcmp("OSKARBIN", header.magic));
    CPPUNIT_ASSERT_EQUAL(header.bin_version, (char)OSKAR_BINARY_FORMAT_VERSION);
    CPPUNIT_ASSERT_EQUAL(header.endian, (char)oskar_endian());
    CPPUNIT_ASSERT_EQUAL(header.size_ptr, (char)sizeof(void*));
    CPPUNIT_ASSERT_EQUAL(header.size_int, (char)sizeof(int));
    CPPUNIT_ASSERT_EQUAL(header.size_long, (char)sizeof(long));
    CPPUNIT_ASSERT_EQUAL(header.size_float, (char)sizeof(float));
    CPPUNIT_ASSERT_EQUAL(header.size_double, (char)sizeof(double));
    CPPUNIT_ASSERT_EQUAL(oskar_binary_header_version(&header),
            (int)OSKAR_VERSION);

    // Create the tag index.
    oskar_BinaryTagIndex idx;
    err = oskar_binary_tag_index_create(&idx, filename);
    CPPUNIT_ASSERT_EQUAL(0, err);
    CPPUNIT_ASSERT_EQUAL(7, idx.num_tags);

    // Read the single numbers back and check values.
    err = oskar_binary_stream_read_int(file, &idx, OSKAR_TAG_NUM_TIMES, 0, 0,
            &a);
    CPPUNIT_ASSERT_EQUAL(0, err);
    CPPUNIT_ASSERT_EQUAL(num_times, a);
    err = oskar_binary_stream_read_int(file, &idx, OSKAR_TAG_NUM_CHANNELS, 0, 0,
            &b);
    CPPUNIT_ASSERT_EQUAL(0, err);
    CPPUNIT_ASSERT_EQUAL(num_channels, b);
    err = oskar_binary_stream_read_int(file, &idx, OSKAR_TAG_NUM_BASELINES, 0, 0,
            &c);
    CPPUNIT_ASSERT_EQUAL(0, err);
    CPPUNIT_ASSERT_EQUAL(num_baselines, c);

    // Read the arrays back and check values.
    {
        std::vector<double> data_double(num_elements_double);
        std::vector<int> data_int(num_elements_int);
        err = oskar_binary_stream_read(file, &idx, OSKAR_TAG_USER, 10, 0,
                OSKAR_INT, size_int, &data_int[0]);
        CPPUNIT_ASSERT_EQUAL(0, err);
        err = oskar_binary_stream_read(file, &idx, OSKAR_TAG_USER, 4, 0,
                OSKAR_DOUBLE, size_double, &data_double[0]);
        CPPUNIT_ASSERT_EQUAL(0, err);
        for (int i = 0; i < num_elements_double; ++i)
            CPPUNIT_ASSERT_DOUBLES_EQUAL(i * 1234.0, data_double[i], 1e-8);
        for (int i = 0; i < num_elements_int; ++i)
            CPPUNIT_ASSERT_EQUAL(i * 10, data_int[i]);
    }
    {
        std::vector<double> data_double(num_elements_double);
        std::vector<int> data_int(num_elements_int);
        err = oskar_binary_stream_read(file, &idx, OSKAR_TAG_USER, 14, 5,
                OSKAR_INT, size_int, &data_int[0]);
        CPPUNIT_ASSERT_EQUAL(0, err);
        err = oskar_binary_stream_read(file, &idx, OSKAR_TAG_USER, 2, 0,
                OSKAR_DOUBLE, size_double, &data_double[0]);
        CPPUNIT_ASSERT_EQUAL(0, err);
        for (int i = 0; i < num_elements_double; ++i)
            CPPUNIT_ASSERT_DOUBLES_EQUAL(i + 1000.0, data_double[i], 1e-8);
        for (int i = 0; i < num_elements_int; ++i)
            CPPUNIT_ASSERT_EQUAL(i * 75, data_int[i]);
    }

    // Look for a tag that isn't there.
    {
        double t;
        err = oskar_binary_stream_read_double(file, &idx,
                OSKAR_TAG_TELESCOPE_ALT_M, 0, 0, &t);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_ERR_BINARY_TAG_NOT_FOUND, err);
    }

    // Close the file.
    fclose(file);

    // Free the tag index.
    oskar_binary_tag_index_free(&idx);

    // Remove the file.
    remove(filename);
}
