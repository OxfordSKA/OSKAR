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

#ifndef OSKAR_GRID_POSITIONS_H_
#define OSKAR_GRID_POSITIONS_H_

/**
 * @file GridPositions.h
 *
 * @brief This file defines functions to generate positions on a 2D grid.
 */

#include <cmath>
#include "math/oskar_Random.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @brief Class used for generating positions on a grid.
 *
 * @details
 * This class provides functions to generate positions on a 2D grid,
 * with optional shape filtering.
 */
class oskar_GridPositions
{
public:
    /// Generates positions on a (randomised) grid within a circle.
    template<typename T>
    static int circular(int seed, T radius, T xs, T ys, T xe, T ye, T* x, T* y);

    /// Generates positions in an Archimedean spiral. ( r = r0 + b * theta )
    template <typename T>
    static void spiralArchimedean(const unsigned n, T * x, T * y,
            const float rMax, const float r0, const float nRevs,
            const float thetaStartDeg);

    /// Generates positions in an Log spiral. ( r = a * exp(b * theta) )
    template <typename T>
    static void spiralLog(const unsigned n, T * x, T * y, const float rMax,
            const float a, const float nRevs, const float thetaStartDeg);
};

/*=============================================================================
 * Static public members
 *---------------------------------------------------------------------------*/

/**
 * @details
 * Generates positions on a (randomised) grid within a circle.
 */
template<typename T>
int oskar_GridPositions::circular(int seed, T radius, T xs, T ys, T xe, T ye,
        T* gx = 0, T* gy = 0)
{
    // Seed the random number generator.
    T r1, r2;
    oskar_Random::gaussian<T>(&r1, &r2, seed);

    int nx = 2.0 * radius / xs;
    int ny = 2.0 * radius / ys;
    int counter = 0;
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            // Compute x and y position on the grid.
            T x = ix * xs - (nx - 1) * xs / 2;
            T y = iy * ys - (ny - 1) * ys / 2;

            // Modify grid position by random numbers.
            oskar_Random::gaussian<T>(&r1, &r2);
            x += xe * r1;
            y += ye * r2;

            // Store if inside filter.
            if (x*x + y*y < radius*radius) {
                if (gx) gx[counter] = x;
                if (gy) gy[counter] = y;
                ++counter;
            }
        }
    }
    return counter;
}


/**
 * Generates positions on a Archimedean spiral.
 *
 * @param n[in]             Number of points to generate.
 * @param x[out]            Coordinate of points in the x direction.
 * @param y[out]            Coordinate of points in the y direction
 * @param rMax[in]          Maximum radius of the spiral (default 1.0)
 * @param r0[in]            Minimum radius of the spiral (default 0.0)
 * @param nRevs[in]         Number of revolutions of the spiral (default 1.0)
 * @param thetaStartDeg[in] Start angle for spiral positions (default 0.0)
 */
template <typename T>
void oskar_GridPositions::spiralArchimedean(const unsigned n, T * x, T * y,
        const float rMax = 1.0f, const float r0 = 0.0f, const float nRevs = 1.0f,
        const float thetaStartDeg = 0.0f)
{
    const float deg2rad = M_PI / 180.0f;
    const float thetaIncDeg = (360.0f * nRevs) / (float)(n - 1);
    const float thetaMaxDeg = thetaStartDeg + thetaIncDeg * (n - 1);
    const float b = (rMax - r0) / (thetaMaxDeg * deg2rad);
    for (unsigned i = 0; i < n; ++i)
    {
        const T thetaRads = (thetaStartDeg + (T)i * thetaIncDeg) * deg2rad;
        const T r = r0 + b * thetaRads;
        x[i] = r * std::cos(thetaRads);
        y[i] = r * std::sin(thetaRads);
    }
}


/**
 * Generates positions in a log spiral.
 *
 * @param n[in]                 Number of points to generate.
 * @param x[out]                x coordinates on the spiral.
 * @param y[out]                y coordinates on the spiral.
 * @param rMax[in]              Maximum radius of the spiral.
 * @param a[in]
 * @param b[in]
 * @param nRevs[in]             Number of revolutions/
 * @param thetaStartDeg[in]     Start position angle.
 */
template <typename T>
void oskar_GridPositions::spiralLog(const unsigned n, T * x, T * y,
        const float rMax = 1.0f, const float a = 0.1f, const float nRevs = 1.0f,
        const float thetaStartDeg = 0.0f)
{
    const float deg2rad = M_PI / 180.0f;
    const float thetaIncDeg = (360.0f * nRevs) / (float)(n - 1);
    const float thetaMaxDeg = thetaStartDeg + thetaIncDeg * (n - 1);
    const float b = std::log(rMax / a) / (thetaMaxDeg * deg2rad);
    for (unsigned i = 0; i < n; ++i)
    {
        T thetaRad = (thetaStartDeg + (T)i * thetaIncDeg) * deg2rad;
        T r = a * std::exp(b * thetaRad);
        x[i] = r * std::cos(thetaRad);
        y[i] = r * std::sin(thetaRad);
    }
}

#endif // OSKAR_GRID_POSITIONS_H_
