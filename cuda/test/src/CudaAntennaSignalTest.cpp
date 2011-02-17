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

#include "cuda/test/CudaAntennaSignalTest.h"
#include "cuda/oskar_cuda_as2hi.h"
#include "math/core/SphericalPositions.h"
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.1415926535
#endif

#define DEG2RAD (M_PI / 180.0)
#define RAD2DEG (180.0 / M_PI)
#define C_0 299792458.0

#define TIMER_ENABLE 1
#include "utility/timer.h"

// Register the test class.
CPPUNIT_TEST_SUITE_REGISTRATION(CudaAntennaSignalTest);

/**
 * @details
 * Sets up the context before running each test method.
 */
void CudaAntennaSignalTest::setUp()
{
}

/**
 * @details
 * Clean up routine called after each test is run.
 */
void CudaAntennaSignalTest::tearDown()
{
}

/**
 * @details
 * Tests antenna signal generation using CUDA.
 */
void CudaAntennaSignalTest::test_method()
{
    // Generate square array of antenna positions.
    const int na = 100;
    const float sep = 0.15; // Antenna separation, metres.
    const float halfArraySize = (na - 1) * sep / 2.0;
    std::vector<float> ax(na * na), ay(na * na); // Antenna (x,y) positions.
    for (int x = 0; x < na; ++x) {
        for (int y = 0; y < na; ++y) {
            int i = y + x * na;
            ax[i] = x * sep - halfArraySize;
            ay[i] = y * sep - halfArraySize;
        }
    }

    // Generate some source positions.
    float centreAz = 0;  // Beam azimuth.
    float centreEl = 50; // Beam elevation.
    SphericalPositions<float> pos (
            centreAz * DEG2RAD, centreEl * DEG2RAD, // Centre.
            30 * DEG2RAD, 30 * DEG2RAD, // Half-widths.
            1 * DEG2RAD, 1 * DEG2RAD); // Spacings.
    unsigned ns = pos.generate(0, 0); // No. of sources.
    std::vector<float> slon(ns), slat(ns);
    pos.generate(&slon[0], &slat[0]);

    // Generate source amplitudes.
    std::vector<float> samp(ns, 1.0);

    // Call CUDA antenna signal generator.
    float freq = 1e9; // Observing frequency, Hertz.
    std::vector<float> signals(na*na * 2); // Antenna signal real & imaginary values.
    TIMER_START
    oskar_cuda_as2hi(na*na, &ax[0], &ay[0], ns, &samp[0], &slon[0], &slat[0],
            2 * M_PI * (freq / C_0), &signals[0]);
    TIMER_STOP("Finished antenna signal generation "
            "(%d antennas, %d sources)", na*na, ns);

    // Write signals data to file.
    FILE* file = fopen("antennaSignal2dHorizontalIsotropic.dat", "w");
    for (unsigned a = 0; a < na*na; ++a) {
        fprintf(file, "%10d%16.4e%16.4e\n", a, signals[2*a], signals[2*a+1]);
    }
    fclose(file);
}
