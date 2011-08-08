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

#include "cuda/test/CudaEquatorialToHorizontalTest.h"
#include "cuda/oskar_cuda_eq2hg.h"
#include "math/oskar_SphericalPositions.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <complex>

#ifndef M_PI
#define M_PI 3.1415926535
#endif

#define DEG2RAD (M_PI / 180.0)
#define RAD2DEG (180.0 / M_PI)
#define C_0 299792458.0

#define TIMER_ENABLE 1
#include "utility/timer.h"

// Register the test class.
CPPUNIT_TEST_SUITE_REGISTRATION(CudaEquatorialToHorizontalTest);

/**
 * @details
 * Sets up the context before running each test method.
 */
void CudaEquatorialToHorizontalTest::setUp()
{
}

/**
 * @details
 * Clean up routine called after each test is run.
 */
void CudaEquatorialToHorizontalTest::tearDown()
{
}

/**
 * @details
 * Tests equatorial to horizontal coordinate transform.
 */
void CudaEquatorialToHorizontalTest::test_separate()
{
    // Generate some equatorial coordinates.
    oskar_SphericalPositions<float> pos (
            0 * DEG2RAD, 90 * DEG2RAD, // Centre.
            30 * DEG2RAD, 30 * DEG2RAD, // Half-widths.
            0.5 * DEG2RAD, 0.5 * DEG2RAD); // Spacings.
    int ns = pos.generate(0, 0); // No. of sources.
    std::vector<float> ra(ns, 0.0), dec(ns, 0.0);
    pos.generate(&ra[0], &dec[0]);

    // Transform to horizontal.
    std::vector<float> az(ns, 0.0), el(ns, 0.0);
    const float lat = 30 * DEG2RAD;
    const float lon = 45 * DEG2RAD;
    float lst = 0 + lon;
    TIMER_START
    oskar_cudaf_eq2hg('s', ns, &ra[0], &dec[0],
            cos(lat), sin(lat), lst, &az[0], &el[0]);
    TIMER_STOP("Finished equatorial to horizontal (separate, %d points)", ns)

    // Write image file.
    FILE* file = fopen("points_separate.txt", "w");
    for (int i = 0; i < ns; ++i) {
        fprintf(file, "%8.3f %8.3f\n", az[i] * RAD2DEG, el[i] * RAD2DEG);
    }
    fclose(file);
}

/**
 * @details
 * Tests equatorial to horizontal coordinate transform.
 */
void CudaEquatorialToHorizontalTest::test_interleaved()
{
    // Generate some equatorial coordinates.
    oskar_SphericalPositions<float> pos (
            0 * DEG2RAD, 90 * DEG2RAD, // Centre.
            30 * DEG2RAD, 30 * DEG2RAD, // Half-widths.
            0.5 * DEG2RAD, 0.5 * DEG2RAD); // Spacings.
    int ns = pos.generate(0, 0); // No. of sources.
    std::vector<float> ra(ns, 0.0), dec(ns, 0.0), radec(2 * ns);
    pos.generate(&ra[0], &dec[0]);

    // Interleave coordinates.
    for (int i = 0; i < ns; ++i) {
        radec[2 * i + 0] = ra[i];
        radec[2 * i + 1] = dec[i];
    }

    // Transform to horizontal.
    std::vector<float> azel(2 * ns, 0.0);
    const float lat = 30 * DEG2RAD;
    const float lon = 45 * DEG2RAD;
    float lst = 0 + lon;
    TIMER_START
    oskar_cudaf_eq2hg('i', ns, &radec[0], 0,
            cos(lat), sin(lat), lst, &azel[0], 0);
    TIMER_STOP("Finished equatorial to horizontal "
            "(interleaved, %d points)", ns)

    // Write image file.
    FILE* file = fopen("points_interleaved.txt", "w");
    for (int i = 0; i < ns; ++i) {
        fprintf(file, "%8.3f %8.3f\n", azel[2 * i + 0] * RAD2DEG,
                 azel[2 * i + 1] * RAD2DEG);
    }
    fclose(file);
}

