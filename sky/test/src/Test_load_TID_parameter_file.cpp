/*
 * Copyright (c) 2013, The University of Oxford
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

#include "sky/test/Test_load_TID_parameter_file.h"

#include "sky/oskar_load_TID_parameter_file.h"
#include "sky/oskar_SettingsIonosphere.h"

#include "utility/oskar_get_error_string.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

/**
 * @details
 */
void Test_load_TID_parameter_file::test_load()
{
    const char* filename = "temp_tid_file.txt";
    double height_km = 300.0;
    double TEC0 = 1.0;
    double comp0[4] = { 1.0, 2.0, 3.0, 4.0 };
    double comp1[4] = { 5.1, 6.1, 7.1, 8.1 };

    // Create a file to test loading.
    {
        FILE* file = fopen(filename, "w");
        fprintf(file, "# some comment\n");
        fprintf(file, "%f\n", height_km);
        fprintf(file, "%f\n", TEC0);
        // amp, speed, theta, wavelength
        fprintf(file, "%f %f %f %f\n", comp0[0], comp0[1], comp0[2], comp0[3]);
        fprintf(file, "%f %f %f %f\n", comp1[0], comp1[1], comp1[2], comp1[3]);
        fflush(file);
        fclose(file);
    }

    // Attempt to read the file!
    int status = OSKAR_SUCCESS;
    oskar_SettingsTIDscreen TID;
    oskar_load_TID_parameter_file(&TID, filename, &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status),
            (int)OSKAR_SUCCESS, status);

    double delta = 1.0e-7;
    CPPUNIT_ASSERT_DOUBLES_EQUAL(height_km, TID.height_km, delta);

    CPPUNIT_ASSERT_EQUAL(2, TID.num_components);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, TID.amp[0], delta);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, TID.speed[0], delta);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, TID.theta[0], delta);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.0, TID.wavelength[0], delta);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(5.1, TID.amp[1], delta);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(6.1, TID.speed[1], delta);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(7.1, TID.theta[1], delta);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(8.1, TID.wavelength[1], delta);

    remove(filename);
}


