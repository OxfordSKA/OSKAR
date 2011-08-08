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

#include "station/oskar_beamPattern.h"
#include "math/oskar_phase.h"
#include <cmath>
#include <vector>

void _generateWeights(const int na, const float* ax, const float* ay,
        float* weights, const float cosBeamEl, const float cosBeamAz,
        const float sinBeamAz, const float k);

void _beamPattern(const int na, const float* ax, const float* ay,
        const float* weights, const int ns, const float* slon, const float* slat,
        const float k, float* image);

/**
 * @details
 * Computes a beam pattern using OpenMP.
 *
 * The function must be supplied with the antenna x- and y-positions, the
 * test source longitude and latitude positions, the beam direction, and
 * the wavenumber.
 *
 * The computed beam pattern is returned in the \p image array, which
 * must be pre-sized to length 2*ns. The values in the \p image array
 * are alternate (real, imag) pairs for each position of the test source.
 *
 * @param[in] na The number of antennas.
 * @param[in] ax The antenna x-positions in metres.
 * @param[in] ay The antenna y-positions in metres.
 * @param[in] ns The number of test source positions.
 * @param[in] slon The longitude coordinates of the test source.
 * @param[in] slat The latitude coordinates of the test source.
 * @param[in] ba The beam azimuth direction in radians
 * @param[in] be The beam elevation direction in radians.
 * @param[in] k The wavenumber (rad / m).
 * @param[out] image The computed beam pattern (see note, above).
 */
void oskar_beamPattern(const int na, const float* ax, const float* ay,
        const int ns, const float* slon, const float* slat,
        const float ba, const float be, const float k,
        float* image)
{
    // Precompute.
    float sinBeamAz = sin(ba);
    float cosBeamAz = cos(ba);
    float cosBeamEl = cos(be);

    // Generate beamforming weights.
    std::vector<float> weights(na * 2);
    _generateWeights(na, ax, ay, &weights[0], cosBeamEl, cosBeamAz,
            sinBeamAz, k);

    // Generate beampattern.
    _beamPattern(na, ax, ay, &weights[0], ns, slon, slat, k, image);
}

/**
 * @details
 */
void _generateWeights(const int na, const float* ax, const float* ay,
        float* weights, const float cosBeamEl, const float cosBeamAz,
        const float sinBeamAz, const float k)
{
    // Loop over antennas.
    for (int a = 0; a < na; ++a) {
        // Compute the geometric phase of the beam direction.
        const float phase = -GEOMETRIC_PHASE_2D_HORIZONTAL(ax[a], ay[a],
                cosBeamEl, sinBeamAz, cosBeamAz, k);
        weights[2*a + 0] = cos(phase) / na; // Normalised real part.
        weights[2*a + 1] = sin(phase) / na; // Normalised imaginary part.
    }
}

/**
 * @details
 */
void _beamPattern(const int na, const float* ax, const float* ay,
        const float* weights, const int ns, const float* slon, const float* slat,
        const float k, float* image)
{
#pragma omp parallel for
    for (int s = 0; s < ns; ++s) {
        // Get the source position.
        const float az = slon[s];
        const float el = slat[s];
        const float cosEl = cos(el);
        const float sinAz = sin(az);
        const float cosAz = cos(az);

        // Loop over all antennas.
        image[2*s + 0] = 0.0;
        image[2*s + 1] = 0.0;
        for (int a = 0; a < na; ++a) {
            // Calculate the geometric phase from the source.
            const float phase = GEOMETRIC_PHASE_2D_HORIZONTAL(ax[a], ay[a],
                    cosEl, sinAz, cosAz, k);
            float signal[2];
            signal[0] = cos(phase);
            signal[1] = sin(phase);

            // Perform complex multiply-accumulate.
            image[2*s + 0] += (signal[0] * weights[2*a + 0] - signal[1] * weights[2*a + 1]); // RE*RE - IM*IM
            image[2*s + 1] += (signal[1] * weights[2*a + 0] + signal[0] * weights[2*a + 1]); // IM*RE + RE*IM
        }
    }
}
