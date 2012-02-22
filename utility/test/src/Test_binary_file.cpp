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
#include "utility/oskar_binary_file_read_header.h"
#include "utility/oskar_binary_file_write_header.h"
#include "utility/oskar_binary_file_write_tag_int.h"
#include "utility/oskar_binary_header_version.h"
#include "utility/oskar_endian.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

void Test_binary_file::test_method()
{
    char filename[] = "cpp_unit_test_binary.dat";
    FILE* file;
    int error;

    // Open a file for binary write.
    file = fopen(filename, "wb");

    // Write the header.
    oskar_binary_file_write_header(file);

    // Write some integers.
    oskar_binary_file_write_tag_int(file, OSKAR_TAG_NUM_TIMES, 0, 0, 65);
    oskar_binary_file_write_tag_int(file, OSKAR_TAG_NUM_BASELINES, 0, 0, 66);
    oskar_binary_file_write_tag_int(file, OSKAR_TAG_NUM_CHANNELS, 0, 0, 67);

    // Close the file.
    fclose(file);

    // Read the header back.
    oskar_BinaryHeader header;
    file = fopen(filename, "rb");
    error = oskar_binary_file_read_header(file, &header);
    CPPUNIT_ASSERT_EQUAL(0, error);
    fclose(file);

    CPPUNIT_ASSERT_EQUAL(0, strcmp("OSKARBIN", header.magic));
    CPPUNIT_ASSERT_EQUAL(header.size_ptr, (char)sizeof(void*));
    CPPUNIT_ASSERT_EQUAL(header.size_size_t, (char)sizeof(size_t));
    CPPUNIT_ASSERT_EQUAL(header.size_int, (char)sizeof(int));
    CPPUNIT_ASSERT_EQUAL(header.size_long, (char)sizeof(long));
    CPPUNIT_ASSERT_EQUAL(header.size_float, (char)sizeof(float));
    CPPUNIT_ASSERT_EQUAL(header.size_double, (char)sizeof(double));
    CPPUNIT_ASSERT_EQUAL(header.endian, (char)oskar_endian());
    CPPUNIT_ASSERT_EQUAL(oskar_binary_header_version(&header),
            (int)OSKAR_VERSION);
}
