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
#include "math/core/Random.h"

/**
 * @brief Class used for generating positions on a grid.
 *
 * @details
 * This class provides functions to generate positions on a 2D grid,
 * with optional shape filtering.
 */
class GridPositions
{
public:
    /// Generates positions on a (randomised) grid within a circle.
    template<typename T>
    static int circular(int seed, T radius, T xs, T ys, T xe, T ye, T* x, T* y);
};

/*=============================================================================
 * Static public members
 *---------------------------------------------------------------------------*/

/**
 * @details
 * Generates positions on a (randomised) grid within a circle.
 */
template<typename T>
int GridPositions::circular(int seed, T radius, T xs, T ys, T xe, T ye,
        T* gx = 0, T* gy = 0)
{
    // Seed the random number generator.
    T r1, r2;
    Random::gaussian<T>(seed, &r1, &r2);

    int nx = 2.0 * radius / xs;
    int ny = 2.0 * radius / ys;
    int counter = 0;
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            // Compute x and y position on the grid.
            T x = ix * xs - (nx - 1) * xs / 2;
            T y = iy * ys - (ny - 1) * ys / 2;

            // Modify grid position by random numbers.
            Random::gaussian<T>(0, &r1, &r2);
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

#endif // OSKAR_GRID_POSITIONS_H_
