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

#include "utility/test/Test_getline.h"
#include "utility/oskar_getline.h"

#include <cstdio>
#include <cstdlib>

void Test_getline::test_method()
{
    int num_lines = 1000, num_chars = 0, i = 0;
    char *line = NULL;
    char temp[1024];
    FILE* file;
    size_t n = 0;

	// Write some dummy data.
    const char* filename = "temp_lines.dat";
    file = fopen(filename, "w");
    if (file == NULL)
    	CPPUNIT_FAIL("Unable to create test file");
    for (i = 0; i < num_lines; ++i)
        fprintf(file, "%.12f,%.12f\n",
        		(double)i/num_lines, (double)i/(10*num_lines));
    fclose(file);

    // Read it in again.
    file = fopen(filename, "r");
    for (i = 0; i < num_lines; ++i)
    {
    	// Read each line.
    	num_chars = oskar_getline(&line, &n, file);

    	// Assert that number of characters per line is correct.
    	CPPUNIT_ASSERT_EQUAL(29, num_chars);

        // Assert that the strings are the same.
        sprintf(temp, "%.12f,%.12f",
        		(double)i/num_lines, (double)i/(10*num_lines));
    	CPPUNIT_ASSERT_EQUAL(0, strcmp(temp, line));
    }
    free(line);
    fclose(file);

    // Cleanup.
    remove(filename);
}

void Test_getline::test_no_final_return_character()
{
    int num_lines = 1000, num_chars = 0, i = 0;
    char *line = NULL;
    char temp[1024];
    FILE* file;
    size_t n = 0;

	// Write some dummy data.
    const char* filename = "temp_lines_no_final_return_character.dat";
    file = fopen(filename, "w");
    if (file == NULL)
    	CPPUNIT_FAIL("Unable to create test file");
    for (i = 0; i < num_lines-1; ++i)
        fprintf(file, "%.12f,%.12f\n",
        		(double)i/num_lines, (double)i/(10*num_lines));
    fprintf(file, "%.12f,%.12f",
    		((double)(num_lines-1))/num_lines,
    		((double)(num_lines-1))/(10*num_lines));
    fclose(file);

    // Read it in again.
    file = fopen(filename, "r");
    i = 0;
    while ((num_chars = oskar_getline(&line, &n, file)) != OSKAR_ERR_EOF)
    {
    	// Assert that number of characters per line is correct.
    	CPPUNIT_ASSERT_EQUAL(29, num_chars);

        // Assert that the strings are the same.
    	sprintf(temp, "%.12f,%.12f",
    			(double)i/num_lines, (double)i/(10*num_lines));
    	CPPUNIT_ASSERT_EQUAL(0, strcmp(temp, line));

    	// Increment line counter.
    	++i;
    }

    // Assert that the number of lines is the same.
	CPPUNIT_ASSERT_EQUAL(num_lines, i);

    free(line);
    fclose(file);

    // Cleanup.
    remove(filename);
}
