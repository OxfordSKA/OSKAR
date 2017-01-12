/*
 * Copyright (c) 2013-2014, The University of Oxford
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

#include "oskar_settings_load_tid_parameter_file.h"
#include "oskar_Settings_old.h"

#include "utility/oskar_get_error_string.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

TEST(TID_parameter_file, load)
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

    // Attempt to read the file.
    int status = 0;
    oskar_SettingsTIDscreen TID;
    oskar_load_tid_parameter_file(&TID, filename, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    EXPECT_DOUBLE_EQ(height_km, TID.height_km);

    EXPECT_EQ(2, TID.num_components);

    EXPECT_DOUBLE_EQ(1.0, TID.amp[0]);
    EXPECT_DOUBLE_EQ(2.0, TID.speed[0]);
    EXPECT_DOUBLE_EQ(3.0, TID.theta[0]);
    EXPECT_DOUBLE_EQ(4.0, TID.wavelength[0]);

    EXPECT_DOUBLE_EQ(5.1, TID.amp[1]);
    EXPECT_DOUBLE_EQ(6.1, TID.speed[1]);
    EXPECT_DOUBLE_EQ(7.1, TID.theta[1]);
    EXPECT_DOUBLE_EQ(8.1, TID.wavelength[1]);

    remove(filename);

    // Free memory in structure.
    free(TID.amp);
    free(TID.speed);
    free(TID.theta);
    free(TID.wavelength);
}
