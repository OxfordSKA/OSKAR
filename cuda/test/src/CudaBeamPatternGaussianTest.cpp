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

#include "cuda/test/CudaBeamPatternGaussianTest.h"
#include "cuda/oskar_cuda_bp2hugg.h"
#include "cuda/oskar_cuda_bp2hcgg.h"
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
CPPUNIT_TEST_SUITE_REGISTRATION(CudaBeamPatternGaussianTest);

/**
 * @details
 * Sets up the context before running each test method.
 */
void CudaBeamPatternGaussianTest::setUp()
{
}

/**
 * @details
 * Clean up routine called after each test is run.
 */
void CudaBeamPatternGaussianTest::tearDown()
{
}

/**
 * @details
 * Tests beam pattern creation using CUDA.
 */
void CudaBeamPatternGaussianTest::test_singleElement()
{
    // Generate square array of antenna positions.
    const int na = 1;
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
    const float aw = 30 * DEG2RAD; // FWHM in radians.
    const float ag = 1.5; // Antenna gain.

    // Generate test source positions.
    float beamAz = 0;  // Beam azimuth.
    float beamEl = 90; // Beam elevation.
    SphericalPositions<float> pos (
            beamAz * DEG2RAD, beamEl * DEG2RAD, // Centre.
            30 * DEG2RAD, 30 * DEG2RAD, // Half-widths.
            0.2 * DEG2RAD, 0.2 * DEG2RAD); // Spacings.
    unsigned ns = pos.generate(0, 0); // No. of sources.
    std::vector<float> slon(ns), slat(ns);
    pos.generate(&slon[0], &slat[0]);

    // Call CUDA beam pattern generator.
    float freq = 1e9; // Observing frequency, Hertz.
    std::vector<float> image(ns * 2); // Beam pattern real & imaginary values.
    TIMER_START
    oskar_cuda_bp2hcgg(na*na, &ax[0], &ay[0], aw, ag, ns, &slon[0],
            &slat[0], beamAz * DEG2RAD, beamEl * DEG2RAD,
            2 * M_PI * (freq / C_0), &image[0]);
    TIMER_STOP("Finished Gaussian beam pattern "
            "(single element, %d points)", ns);

    // Write image data to file.
    FILE* file = fopen("antPatternCommonGaussian2dHorizontalGeometricNorm.dat", "w");
    for (unsigned s = 0; s < ns; ++s) {
        fprintf(file, "%12.3f%12.3f%16.4e%16.4e\n",
                slon[s] * RAD2DEG, slat[s] * RAD2DEG, image[2*s], image[2*s+1]);
    }
    fclose(file);
}

/**
 * @details
 * Tests beam pattern creation using CUDA.
 */
void CudaBeamPatternGaussianTest::test_commonNormalised()
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
    const float aw = 30 * DEG2RAD; // FWHM in radians.
    const float ag = 1.5; // Antenna gain.

    // Generate test source positions.
    float beamAz = 0;  // Beam azimuth.
    float beamEl = 75; // Beam elevation.
    SphericalPositions<float> pos (
            beamAz * DEG2RAD, beamEl * DEG2RAD, // Centre.
            30 * DEG2RAD, 30 * DEG2RAD, // Half-widths.
            0.2 * DEG2RAD, 0.2 * DEG2RAD); // Spacings.
    unsigned ns = pos.generate(0, 0); // No. of sources.
    std::vector<float> slon(ns), slat(ns);
    pos.generate(&slon[0], &slat[0]);

    // Call CUDA beam pattern generator.
    float freq = 1e9; // Observing frequency, Hertz.
    std::vector<float> image(ns * 2); // Beam pattern real & imaginary values.
    TIMER_START
    oskar_cuda_bp2hcgg(na*na, &ax[0], &ay[0], aw, ag, ns, &slon[0],
            &slat[0], beamAz * DEG2RAD, beamEl * DEG2RAD,
            2 * M_PI * (freq / C_0), &image[0]);
    TIMER_STOP("Finished common Gaussian beam pattern "
            "(%d element regular array, %d points)", na*na, ns);

    // Write image data to file.
    FILE* file = fopen("beamPatternCommonGaussian2dHorizontalGeometricNorm.dat", "w");
    for (unsigned s = 0; s < ns; ++s) {
        fprintf(file, "%12.3f%12.3f%16.4e%16.4e\n",
                slon[s] * RAD2DEG, slat[s] * RAD2DEG, image[2*s], image[2*s+1]);
    }
    fclose(file);
}

/**
 * @details
 * Tests beam pattern creation using CUDA.
 */
void CudaBeamPatternGaussianTest::test_commonNormalised1d()
{
    // Generate linear array of antenna positions.
    const int na = 100;
    const float sep = 0.15; // Antenna separation, metres.
    const float halfArraySize = (na - 1) * sep / 2.0;
    std::vector<float> ax(na), ay(na); // Antenna (x,y) positions.
    for (int x = 0; x < na; ++x) {
        ax[x] = x * sep - halfArraySize;
        ay[x] = 0;
    }
    const float aw = 30 * DEG2RAD; // FWHM in radians.
    const float ag = 1.5; // Antenna gain.

    // Generate test source positions.
    float beamAz = 0;  // Beam azimuth.
    float beamEl = 90; // Beam elevation.
    SphericalPositions<float> pos (
            beamAz * DEG2RAD, beamEl * DEG2RAD, // Centre.
            30 * DEG2RAD, 0 * DEG2RAD, // Half-widths.
            0.2 * DEG2RAD, 0.2 * DEG2RAD); // Spacings.
    unsigned ns = pos.generate(0, 0); // No. of sources.
    std::vector<float> slon(ns), slat(ns);
    pos.generate(&slon[0], &slat[0]);

    // Call CUDA beam pattern generator.
    float freq = 1e9; // Observing frequency, Hertz.
    std::vector<float> image(ns * 2); // Beam pattern real & imaginary values.
    TIMER_START
    oskar_cuda_bp2hcgg(na, &ax[0], &ay[0], aw, ag, ns, &slon[0],
            &slat[0], beamAz * DEG2RAD, beamEl * DEG2RAD,
            2 * M_PI * (freq / C_0), &image[0]);
    TIMER_STOP("Finished common Gaussian beam pattern "
            "(%d element regular array, %d points)", na, ns);

    // Write image data to file.
    FILE* file = fopen("beamPatternCommonGaussian1dHorizontalGeometricNorm.dat", "w");
    for (unsigned s = 0; s < ns; ++s) {
        fprintf(file, "%12.3f%12.3f%16.4e%16.4e\n",
                slon[s] * RAD2DEG, slat[s] * RAD2DEG, image[2*s], image[2*s+1]);
    }
    fclose(file);
}

/**
 * @details
 * Tests beam pattern creation using CUDA.
 */
void CudaBeamPatternGaussianTest::test_uniqueNormalised()
{
    // Generate square array of antenna positions.
    const int na = 100;
    const float sep = 0.15; // Antenna separation, metres.
    const float halfArraySize = (na - 1) * sep / 2.0;
    std::vector<float> ax(na * na), ay(na * na); // Antenna (x,y) positions.
    std::vector<float> aw(na * na), ag(na * na); // Antenna beam parameters.
    for (int x = 0; x < na; ++x) {
        for (int y = 0; y < na; ++y) {
            int i = y + x * na;
            ax[i] = x * sep - halfArraySize;
            ay[i] = y * sep - halfArraySize;
            aw[i] = 30 * DEG2RAD; // FWHM in radians.
            ag[i] = 1.5; // Antenna gain.
        }
    }

    // Generate test source positions.
    float beamAz = 0;  // Beam azimuth.
    float beamEl = 75; // Beam elevation.
    SphericalPositions<float> pos (
            beamAz * DEG2RAD, beamEl * DEG2RAD, // Centre.
            30 * DEG2RAD, 30 * DEG2RAD, // Half-widths.
            0.2 * DEG2RAD, 0.2 * DEG2RAD); // Spacings.
    unsigned ns = pos.generate(0, 0); // No. of sources.
    std::vector<float> slon(ns), slat(ns);
    pos.generate(&slon[0], &slat[0]);

    // Call CUDA beam pattern generator.
    float freq = 1e9; // Observing frequency, Hertz.
    std::vector<float> image(ns * 2); // Beam pattern real & imaginary values.
    TIMER_START
    oskar_cuda_bp2hugg(na*na, &ax[0], &ay[0], &aw[0], &ag[0], ns, &slon[0],
            &slat[0], beamAz * DEG2RAD, beamEl * DEG2RAD,
            2 * M_PI * (freq / C_0), &image[0]);
    TIMER_STOP("Finished unique Gaussian beam pattern "
            "(%d element regular array, %d points)", na*na, ns);

    // Write image data to file.
    FILE* file = fopen("beamPatternUniqueGaussian2dHorizontalGeometricNorm.dat", "w");
    for (unsigned s = 0; s < ns; ++s) {
        fprintf(file, "%12.3f%12.3f%16.4e%16.4e\n",
                slon[s] * RAD2DEG, slat[s] * RAD2DEG, image[2*s], image[2*s+1]);
    }
    fclose(file);
}

/**
 * @details
 * Tests beam pattern creation using CUDA.
 */
void CudaBeamPatternGaussianTest::test_uniqueNormalised1d()
{
    // Generate linear array of antenna positions.
    const int na = 100;
    const float sep = 0.15; // Antenna separation, metres.
    const float halfArraySize = (na - 1) * sep / 2.0;
    std::vector<float> ax(na), ay(na); // Antenna (x,y) positions.
    std::vector<float> aw(na), ag(na); // Antenna beam parameters.
    for (int x = 0; x < na; ++x) {
        ax[x] = x * sep - halfArraySize;
        ay[x] = 0;
        aw[x] = 30 * DEG2RAD; // FWHM in radians.
        ag[x] = 1.5; // Antenna gain.
    }

    // Generate test source positions.
    float beamAz = 0;  // Beam azimuth.
    float beamEl = 90; // Beam elevation.
    SphericalPositions<float> pos (
            beamAz * DEG2RAD, beamEl * DEG2RAD, // Centre.
            30 * DEG2RAD, 0 * DEG2RAD, // Half-widths.
            0.2 * DEG2RAD, 0.2 * DEG2RAD); // Spacings.
    unsigned ns = pos.generate(0, 0); // No. of sources.
    std::vector<float> slon(ns), slat(ns);
    pos.generate(&slon[0], &slat[0]);

    // Call CUDA beam pattern generator.
    float freq = 1e9; // Observing frequency, Hertz.
    std::vector<float> image(ns * 2); // Beam pattern real & imaginary values.
    TIMER_START
    oskar_cuda_bp2hugg(na, &ax[0], &ay[0], &aw[0], &ag[0], ns, &slon[0],
            &slat[0], beamAz * DEG2RAD, beamEl * DEG2RAD,
            2 * M_PI * (freq / C_0), &image[0]);
    TIMER_STOP("Finished unique Gaussian beam pattern "
            "(%d element regular array, %d points)", na, ns);

    // Write image data to file.
    FILE* file = fopen("beamPatternUniqueGaussian1dHorizontalGeometricNorm.dat", "w");
    for (unsigned s = 0; s < ns; ++s) {
        fprintf(file, "%12.3f%12.3f%16.4e%16.4e\n",
                slon[s] * RAD2DEG, slat[s] * RAD2DEG, image[2*s], image[2*s+1]);
    }
    fclose(file);
}
