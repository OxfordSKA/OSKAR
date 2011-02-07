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
        T* gx, T* gy)
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
