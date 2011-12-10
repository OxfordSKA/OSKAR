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

#include "math/test/Test_SphericalPositions.h"
#include "math/oskar_SphericalPositions.h"
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DEG2RAD (M_PI / 180.0)



/**
 * @details
 * Sets up the context before running each test method.
 */
void Test_SphericalPositions::setUp()
{
}

/**
 * @details
 * Clean up routine called after each test is run.
 */
void Test_SphericalPositions::tearDown()
{
}

/**
 * @details
 * Tests generation of points using parameters to generate a 13 x 13 grid.
 */
void Test_SphericalPositions::test_generate13x13()
{
    oskar_SphericalPositions<float> posGen(0, 90 * DEG2RAD,
            2 * DEG2RAD, 2 * DEG2RAD, // Half-widths
            0.3 * DEG2RAD, 0.3 * DEG2RAD);

    // Should have a 13 x 13 point grid.
    unsigned points = posGen.generate(0 ,0);
    CPPUNIT_ASSERT_EQUAL(169u, points);
    std::vector<float> longitudes(points);
    std::vector<float> latitudes(points);
    posGen.generate(&longitudes[0], &latitudes[0]);
}
