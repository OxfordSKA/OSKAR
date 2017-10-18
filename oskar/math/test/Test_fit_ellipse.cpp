/*
 * Copyright (c) 2012-2016, The University of Oxford
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

#include "math/oskar_fit_ellipse.h"
#include "utility/oskar_get_error_string.h"

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>

TEST(fit_ellipse, test1)
{
    int num_points = 7, status = 0;
    double maj_d = 0.0, min_d = 0.0, pa_d = 0.0;
    float maj_f = 0.0, min_f = 0.0, pa_f = 0.0;

    // Test double precision.
    {
        std::vector<double> x(num_points), y(num_points);
        std::vector<double> work1(5 * num_points), work2(5 * num_points);
        x[0] = -0.1686;
        x[1] = -0.0921;
        x[2] =  0.0765;
        x[3] =  0.1686;
        x[4] =  0.0921;
        x[5] = -0.0765;
        x[6] = -0.1686;

        y[0] =  0.7282;
        y[1] =  0.6994;
        y[2] =  0.6675;
        y[3] =  0.6643;
        y[4] =  0.7088;
        y[5] =  0.7407;
        y[6] =  0.7282;

        oskar_fit_ellipse_d(&maj_d, &min_d, &pa_d, num_points, &x[0], &y[0],
                &work1[0], &work2[0], &status);
        EXPECT_EQ(0, status) << oskar_get_error_string(status);
    }

    // Test single precision.
    {
        std::vector<float> x(num_points), y(num_points);
        std::vector<float> work1(5 * num_points), work2(5 * num_points);
        x[0] = -0.1686f;
        x[1] = -0.0921f;
        x[2] =  0.0765f;
        x[3] =  0.1686f;
        x[4] =  0.0921f;
        x[5] = -0.0765f;
        x[6] = -0.1686f;

        y[0] =  0.7282f;
        y[1] =  0.6994f;
        y[2] =  0.6675f;
        y[3] =  0.6643f;
        y[4] =  0.7088f;
        y[5] =  0.7407f;
        y[6] =  0.7282f;

        oskar_fit_ellipse_f(&maj_f, &min_f, &pa_f, num_points, &x[0], &y[0],
                &work1[0], &work2[0], &status);
        EXPECT_EQ(0, status) << oskar_get_error_string(status);
    }

    // Compare results.
    EXPECT_NEAR(maj_d, 0.3608619735, 1e-9);
    EXPECT_NEAR(min_d, 0.0494223702, 1e-9);
    EXPECT_NEAR(pa_d, -1.3865537748, 1e-9);
    EXPECT_NEAR(maj_d, maj_f, 1e-5);
    EXPECT_NEAR(min_d, min_f, 1e-5);
    EXPECT_NEAR(pa_d, pa_f, 1e-5);
}
