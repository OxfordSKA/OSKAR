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

#include "cuda/test/Imager2dDftTest.h"
#include "cuda/imager2dDft.h"
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
CPPUNIT_TEST_SUITE_REGISTRATION(Imager2dDftTest);

/**
 * @details
 * Sets up the context before running each test method.
 */
void Imager2dDftTest::setUp()
{
}

/**
 * @details
 * Clean up routine called after each test is run.
 */
void Imager2dDftTest::tearDown()
{
}

/**
 * @details
 * Tests 2D DFT CUDA imager.
 */
void Imager2dDftTest::test()
{
    // Set up some visibilities.
    const int nv = 1;
    std::vector<float> vis(2 * nv, 0.0), u(nv, 0.0), v(nv, 0.0);

    u[0] = 1.0f;
    v[0] = 1.0f;
    vis[0] = 1.0f; // real.
    vis[1] = 1.0f; // imag.

    // Image the visibilities.
    int nl = 32;
    int nm = 32;
    float dl = 1.0 / nl;
    float dm = 1.0 / nm;
    float sl = M_PI;
    float sm = M_PI;
    std::vector<float> image(nl * nm);
    TIMER_START
    imager2dDft(nv, &u[0], &v[0], &vis[0], nl, nm, dl, dm, sl, sm, &image[0]);
    TIMER_STOP("Finished DFT imager (%d x %d, %d visibilities)", nl, nm, nv)

    // Write image file.
    FILE* file = fopen("output.txt", "w");
    for (int j = 0; j < nm; ++j) {
        for (int i = 0; i < nl; ++i) {
            fprintf(file, "%.5e ", image[i + j * nl]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

/**
 * @details
 * Tests 2D DFT CUDA imager.
 */
void Imager2dDftTest::test_large()
{
    // Set up some visibilities.
    const int nv = 100000;
    std::vector<float> vis(2 * nv, 0.0), u(nv, 0.0), v(nv, 0.0);

    u[0] = 1.0f;
    v[0] = 1.0f;
    vis[0] = 1.0f; // real.
    vis[1] = 1.0f; // imag.

    // Image the visibilities.
    int nl = 1024;
    int nm = 1024;
    float dl = 1.0 / nl;
    float dm = 1.0 / nm;
    float sl = M_PI;
    float sm = M_PI;
    std::vector<float> image(nl * nm);
    TIMER_START
    imager2dDft(nv, &u[0], &v[0], &vis[0], nl, nm, dl, dm, sl, sm, &image[0]);
    TIMER_STOP("Finished DFT imager (%d x %d, %d visibilities)", nl, nm, nv)

    // Write image file.
    FILE* file = fopen("output_large.txt", "w");
    for (int j = 0; j < nm; ++j) {
        for (int i = 0; i < nl; ++i) {
            fprintf(file, "%.5e ", image[i + j * nl]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}
