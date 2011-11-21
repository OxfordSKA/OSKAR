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

#include "utility/test/Test_getline.h"
#include "utility/oskar_getline.h"

#include <cstdio>
#include <cstdlib>

void Test_getline::test_method()
{
	// Write some dummy data.
    const char* filename = "temp_lines.dat";
    FILE* file = fopen(filename, "w");
    if (file == NULL)
    	CPPUNIT_FAIL("Unable to create test file");
    int num_coords = 1000;
    for (int i = 0; i < num_coords; ++i)
        fprintf(file, "%.12f,%.12f\n",
        		(double)i/num_coords, (double)i/(10*num_coords));
    fclose(file);

    // Read it in again.
    char* line = NULL;
    size_t n = 0;
    file = fopen(filename, "r");
    char temp[1024];
    for (int i = 0; i < num_coords; ++i)
    {
    	// Read each line.
    	int num_chars = oskar_getline(&line, &n, file);

    	// Assert that number of characters per line is correct.
    	CPPUNIT_ASSERT_EQUAL(30, num_chars);
        sprintf(temp, "%.12f,%.12f\n",
        		(double)i/num_coords, (double)i/(10*num_coords));

        // Assert that the strings are the same.
    	int flag = strcmp(temp, line);
    	CPPUNIT_ASSERT_EQUAL(0, flag);
    }
    free(line);
    fclose(file);

    // Cleanup.
    remove(filename);
}
