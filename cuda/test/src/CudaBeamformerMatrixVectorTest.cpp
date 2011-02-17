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

#include "cuda/test/CudaBeamformerMatrixVectorTest.h"
#include "cuda/oskar_cuda_bfmv.h"
#include "cuda/oskar_cuda_bf2hig.h"
#include "math/core/SphericalPositions.h"
#include <cstdio>
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
CPPUNIT_TEST_SUITE_REGISTRATION(CudaBeamformerMatrixVectorTest);

/**
 * @details
 * Sets up the context before running each test method.
 */
void CudaBeamformerMatrixVectorTest::setUp()
{
}

/**
 * @details
 * Clean up routine called after each test is run.
 */
void CudaBeamformerMatrixVectorTest::tearDown()
{
}

/**
 * @details
 * Tests antenna signal generation using CUDA.
 */
void CudaBeamformerMatrixVectorTest::test_basicMatrixVector()
{
    unsigned na = 3;
    unsigned nb = 2;

    // Allocate memory for signals, weights and beams.
    std::vector<float> signals(na * 2, 0.0), beams(nb * 2, 0.0);
    std::vector<float> weights(na * nb * 2, 0.0);

    // Fill signal and weights arrays.
    for (unsigned i = 0; i < na * 2; i += 2) signals[i] = i + 1;
    for (unsigned i = 0; i < na * nb * 2; i += 2) weights[i] = i + 2;

    // Perform matrix-matrix multiply.
    oskar_cuda_bfmv(na, nb, &signals[0], &weights[0], &beams[0]);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(44.0, beams[0], 0.01);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(98.0, beams[2], 0.01);

    printf("\n");
    for (unsigned i = 0, b = 0; b < nb; i += 2, b++) {
        printf("Beam %5d : (%4.1f, %4.1f)\n", b, beams[i], beams[i + 1]);
    }
}

/**
 * @details
 * Tests antenna signal generation using CUDA.
 */
void CudaBeamformerMatrixVectorTest::test_method()
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
    SphericalPositions<float> posSrc (
            centreAz * DEG2RAD, centreEl * DEG2RAD, // Centre.
            20 * DEG2RAD, 20 * DEG2RAD, // Half-widths.
            10 * DEG2RAD, 10 * DEG2RAD); // Spacings.
    unsigned ns = posSrc.generate(0, 0); // No. of sources.
    std::vector<float> slon(ns), slat(ns);
    posSrc.generate(&slon[0], &slat[0]);

    // Generate source amplitudes.
    std::vector<float> samp(ns, 1.0);

    // Generate some beam positions.
    SphericalPositions<float> posBeam (
            centreAz * DEG2RAD, centreEl * DEG2RAD, // Centre.
            30 * DEG2RAD, 30 * DEG2RAD, // Half-widths.
            0.2 * DEG2RAD, 0.2 * DEG2RAD); // Spacings.
    unsigned nb = posBeam.generate(0, 0); // No. of beams.
    std::vector<float> blon(nb), blat(nb);
    posBeam.generate(&blon[0], &blat[0]);

    // Call CUDA beamformer.
    float freq = 1e9; // Observing frequency, Hertz.
    std::vector<float> beams(nb * 2); // Beam real & imaginary values.
    TIMER_START
    oskar_cuda_bf2hig(na*na, &ax[0], &ay[0], ns, &samp[0],
            &slon[0], &slat[0], nb, &blon[0], &blat[0],
            2 * M_PI * (freq / C_0), &beams[0]);
    TIMER_STOP("Finished beamforming "
            "(%d antennas, %d sources, %d beams)", na*na, ns, nb);

    // Write beam data to file.
    FILE* file = fopen("beams.dat", "w");
    for (unsigned b = 0; b < nb; ++b) {
        fprintf(file, "%12.3f%12.3f%16.4e%16.4e\n",
                blon[b] * RAD2DEG, blat[b] * RAD2DEG, beams[2*b], beams[2*b+1]);
    }
    fclose(file);
}
