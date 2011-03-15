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

#include "cuda/test/CudaHierarchicalBeamPatternTest.h"
#include "cuda/oskar_cuda_hbp2hig.h"
#include "math/core/SphericalPositions.h"
#include "math/core/GridPositions.h"
#include "math/core/Matrix3.h"
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
CPPUNIT_TEST_SUITE_REGISTRATION(CudaHierarchicalBeamPatternTest);

/**
 * @details
 * Sets up the context before running each test method.
 */
void CudaHierarchicalBeamPatternTest::setUp()
{
}

/**
 * @details
 * Clean up routine called after each test is run.
 */
void CudaHierarchicalBeamPatternTest::tearDown()
{
}

/**
 * @details
 * Tests beam pattern creation using CUDA.
 */
void CudaHierarchicalBeamPatternTest::test_regular()
{
    // Square root of number of tiles, number of antennas per tile.
    const int nt = 8;
    const int na = 16;

    // Separations.
    const float asep = 0.15; // Antenna separation, metres.
    const float tsep = 0.0;  // Extra tile separation, metres.

    // Define storage for antenna and tile coordinates.
    const int nt2 = nt * nt; // Number of tiles per station.
    const int na2 = na * na; // Number of antennas per tile.
    const int nas = na2 * nt2; // Total number of antennas per station.
    std::vector<float> ax(nas), ay(nas), tx(nt2), ty(nt2);

    // Define storage for number of antennas per tile.
    std::vector<int> nav(nt2, na2); // Length nt2, each containing na2.

    // Compute tile and station sizes.
    const float halfTileSize = (na - 1) * asep / 2.0;
    const float halfStationSize = (nt - 1) * (na * asep + tsep) / 2.0;

    // Loop over tiles.
    for (int itx = 0; itx < nt; ++itx) {
        for (int ity = 0; ity < nt; ++ity) {

            // Get tile index.
            const int t = ity + itx * nt;

            // Store tile coordinates.
            tx[t] = itx * (na * asep + tsep) - halfStationSize;
            ty[t] = ity * (na * asep + tsep) - halfStationSize;

            // Loop over antennas per tile.
            for (int iax = 0; iax < na; ++iax) {
                for (int iay = 0; iay < na; ++iay) {
                    const int a = (t * na2) + iay + iax * na;
                    ax[a] = iax * asep - halfTileSize;
                    ay[a] = iay * asep - halfTileSize;
                }
            }
        }
    }

//    for (int iax = 0; iax < na; ++iax) {
//        for (int iay = 0; iay < na; ++iay) {
//            const int a = (3 * na2) + iay + iax * na;
//            printf("ax, ay = %.3f, %.3f\n", ax[a], ay[a]);
//        }
//    }

//    for (int iax = 0; iax < nt; ++iax) {
//        for (int iay = 0; iay < nt; ++iay) {
//            const int a = iay + iax * nt;
//            printf("tx, ty = %.3f, %.3f\n", tx[a], ty[a]);
//        }
//    }

    // Define beam directions.
    float stationBeamAz = 2 * DEG2RAD;  // Beam azimuth.
    float stationBeamEl = 48 * DEG2RAD; // Beam elevation.
    float tileBeamAz = 0 * DEG2RAD;  // Beam azimuth.
    float tileBeamEl = 50 * DEG2RAD; // Beam elevation.

    // Generate test source positions.
    SphericalPositions<float> pos (
            stationBeamAz, stationBeamEl, // Centre.
            30 * DEG2RAD, 30 * DEG2RAD, // Half-widths.
            0.2 * DEG2RAD, 0.2 * DEG2RAD); // Spacings.
    unsigned ns = pos.generate(0, 0); // No. of sources.
    std::vector<float> sa(ns), se(ns);
    pos.generate(&sa[0], &se[0]);

    // Call CUDA beam pattern generator.
    const float freq = 1e9; // Observing frequency, Hertz.
    const float k = 2 * M_PI * (freq / C_0); // Wavenumber.
    std::vector<float> image(ns * 2); // Beam pattern real & imaginary values.
    TIMER_START
    oskar_cuda_hbp2hig(nt2, &nav[0], &ax[0], &ay[0], &tx[0], &ty[0],
            ns, &sa[0], &se[0], tileBeamAz, tileBeamEl,
            stationBeamAz, stationBeamEl, k, &image[0]);
    TIMER_STOP("Finished hierarchical pattern (%d element regular array, %d points)",
            nas, ns);

    // Write image data to file.
    FILE* file = fopen("hierarchicalBeamPattern2dHorizontalGeometricRegular.dat", "w");
    for (unsigned s = 0; s < ns; ++s) {
        fprintf(file, "%12.3f%12.3f%16.4e%16.4e\n",
                sa[s] * RAD2DEG, se[s] * RAD2DEG, image[2*s], image[2*s+1]);
    }
    fclose(file);
}
