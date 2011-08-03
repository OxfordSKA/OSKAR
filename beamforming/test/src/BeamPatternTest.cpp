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

#include "beamforming/test/BeamPatternTest.h"
#include "beamforming/oskar_beamPattern.h"
#include "math/SphericalPositions.h"

#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DEG2RAD (M_PI / 180.0)
#define RAD2DEG (180.0 / M_PI)
#define C_0 299792458.0

#define TIMER_ENABLE 1
#include "utility/timer.h"



/**
 * @details
 * Sets up the context before running each test method.
 */
void BeamPatternTest::setUp()
{
}

/**
 * @details
 * Clean up routine called after each test is run.
 */
void BeamPatternTest::tearDown()
{
}

/**
 * @details
 * Tests beam pattern creation using CUDA.
 */
void BeamPatternTest::test_method()
{
    // ==== Generate square array of antenna positions.
    const int num_antennas = 100;
    const float antenna_sep_metres = 0.15;
    const float halfArraySize = (num_antennas - 1) * antenna_sep_metres / 2.0;
    std::vector<float> antenna_x(num_antennas * num_antennas);
    std::vector<float> antenna_y(num_antennas * num_antennas);
    for (int x = 0; x < num_antennas; ++x)
    {
        for (int y = 0; y < num_antennas; ++y)
        {
            int i = y + x * num_antennas;
            antenna_x[i] = x * antenna_sep_metres - halfArraySize;
            antenna_y[i] = y * antenna_sep_metres - halfArraySize;
        }
    }


    // ==== Generate test source positions.
    float beamAz = 0;  // Beam azimuth.
    float beamEl = 50; // Beam elevation.
    SphericalPositions<float> pos(
            beamAz * DEG2RAD, beamEl * DEG2RAD, // Centre.
            30 * DEG2RAD, 30 * DEG2RAD, // Half-widths.
            0.2 * DEG2RAD, 0.2 * DEG2RAD); // Spacings.
    unsigned num_sources = pos.generate(0, 0); // No. of sources.
    std::vector<float> slon(num_sources), slat(num_sources);
    pos.generate(&slon[0], &slat[0]);


    // ==== Call beam pattern generator.
    float freq = 1e9; // Observing frequency, Hertz.
    std::vector<float> image(num_sources * 2); // Beam pattern real & imaginary values.
    TIMER_START
    oskar_beamPattern(num_antennas * num_antennas, &antenna_x[0],
            &antenna_y[0], num_sources, &slon[0], &slat[0],
            beamAz * DEG2RAD, beamEl * DEG2RAD, 2 * M_PI * (freq / C_0),
            &image[0]);
    TIMER_STOP("Finished beam pattern");


    // ==== Write image data to file.
    FILE* file = fopen("beamPattern.dat", "w");
    for (unsigned s = 0; s < num_sources; ++s)
    {
        fprintf(file, "%12.3f%12.3f%16.4e%16.4e\n",
                slon[s] * RAD2DEG, slat[s] * RAD2DEG, image[2*s], image[2*s+1]);
    }
    fclose(file);
}
