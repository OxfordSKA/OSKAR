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

#ifndef OSKAR_INTERP_BILINEAR_H_
#define OSKAR_INTERP_BILINEAR_H_

#include <cmath>

/**
 * @brief Class used for bilinear interpolation.
 *
 * @details
 *
 */
template<typename T>
class oskar_InterpBilinear
{
public:
    /// Constructs a data structure for bilinear interpolation.
    oskar_InterpBilinear(const T* grid, int nx, int ny,
            T xmin, T xmax, T ymin, T ymax);

    /// Return the value at the given position by interpolation.
    T interpolate(T x, T y) const;

private:
    /// Perform bilinear interpolation.
    static T _interpolate(T x1, T x2, T y1, T y2,
            T f11, T f12, T f21, T f22, T x, T y, T dinv);

private:
    const T* _grid; ///< Pointer to the grid of data to interpolate.
    int _nx;        ///< Number of points in x-dimension.
    int _ny;        ///< Number of points in y-dimension.
    T _xmin;        ///< Minimum x-value.
    T _xmax;        ///< Maximum x-value.
    T _ymin;        ///< Minimum y-value.
    T _ymax;        ///< Maximum y-value.
    T _xrange;      ///< Range in x.
    T _yrange;      ///< Range in y.
    T _xspace;      ///< Spacing in x.
    T _yspace;      ///< Spacing in y.
    T _xspaceinv;   ///< Inverse spacing in x.
    T _yspaceinv;   ///< Inverse spacing in y.
    T _dinv;        ///< The quantity 1.0 / (xspace * yspace).
};

/**
 * @brief Constructs a data structure for bilinear interpolation.
 *
 * @details
 * Pre-computes and stores the grid parameters required for bilinear
 * interpolation. The data grid is row-major (C-ordered) and increases
 * first along the x dimension.
 *
 * @param[in] grid The data grid over which to interpolate.
 * @param[in] nx Number of points in x.
 * @param[in] ny Number of points in y.
 * @param[in] xmin Minimum x value.
 * @param[in] xmax Maximum x value.
 * @param[in] ymin Minimum y value.
 * @param[in] ymax Maximum y value.
 */
template<typename T>
oskar_InterpBilinear::oskar_InterpBilinear(const T* grid, int nx, int ny,
        T xmin, T xmax, T ymin, T ymax)
{
    _grid = grid;
    _nx = nx;
    _ny = ny;
    _xmin = (xmin < xmax) ? xmin : xmax;
    _xmax = (xmin > xmax) ? xmin : xmax;
    _ymin = (ymin < ymax) ? ymin : ymax;
    _ymax = (ymin > ymax) ? ymin : ymax;
    _xrange = std::fabs(xmax - xmin);
    _yrange = std::fabs(ymax - ymin);
    _xspace = _xrange / (nx - 1);
    _yspace = _yrange / (ny - 1);
    _xspaceinv = 1.0 / _xspace;
    _yspaceinv = 1.0 / _yspace;
    _dinv = 1.0 / (_xspace * _yspace);
}

/**
 * @details
 * Performs bilinear interpolation on the grid to evaluate the data
 * at the required position.
 *
 * @param[in] x     The required x coordinate.
 * @param[in] y     The required y coordinate.
 */
template<typename T>
T oskar_InterpBilinear::interpolate(T x, T y) const
{
    // Find indices of grid points around x and y.
    int xi1 = (x - _xmin) * _xspaceinv;
    int yi1 = (y - _ymin) * _yspaceinv;
    int xi2 = xi1 + 1; //(xi1 < _nx - 1) ? xi1 + 1 : 0; //_nx - 1;
    int yi2 = yi1 + 1; //(yi1 < _ny - 1) ? yi1 + 1 : 0; //_ny - 1;

    // Find scale values at grid points.
    T x1 = xi1 * _xspace + _xmin;
    T y1 = yi1 * _yspace + _ymin;
    T x2 = xi2 * _xspace + _xmin;
    T y2 = yi2 * _yspace + _ymin;

    // Find data values at grid points.
    if (xi2 >= _nx) xi2 = 0;
    if (yi2 >= _ny) yi2 = 0;
    int yi1offset = yi1 * _nx;
    int yi2offset = yi2 * _nx;
    T f11 = data[xi1 + yi1offset];
    T f12 = data[xi1 + yi2offset];
    T f21 = data[xi2 + yi1offset];
    T f22 = data[xi2 + yi2offset];

    // Perform bilinear interpolation.
    return _interpolate(x1, x2, y1, y2, f11, f12, f21, f22, x, y, _dinv);
}


/**
 * @details
 * Performs bilinear interpolation from the four points
 * at (x1, y1), (x1, y2), (x2, y1), (x2, y2).
 *
 * @param[in] x1   The first x-coordinate.
 * @param[in] x2   The second x-coordinate.
 * @param[in] y1   The first y-coordinate.
 * @param[in] y2   The second y-coordinate.
 * @param[in] f11  The value of the function at (x1, y1).
 * @param[in] f12  The value of the function at (x1, y2).
 * @param[in] f21  The value of the function at (x2, y1).
 * @param[in] f22  The value of the function at (x2, y2).
 * @param[in] x    The x-coordinate of the point to return.
 * @param[in] y    The y-coordinate of the point to return.
 * @param[in] dinv The quantity 1.0 / ((x2 - x1) * (y2 - y1)).
 *
 * @return The interpolated value at the point (x, y).
 */
template<typename T>
static T oskar_InterpBilinear::_interpolate(T x1, T x2, T y1, T y2,
        T f11, T f12, T f21, T f22, T x, T y, T dinv)
{
    T x1d = x - x1;
    T y1d = y - y1;
    T x2d = x2 - x;
    T y2d = y2 - y;
    T p11 = f11 * x2d * y2d;
    T p21 = f21 * x1d * y2d;
    T p12 = f12 * x2d * y1d;
    T p22 = f22 * x1d * y1d;
    T s = p11 + p21 + p12 + p22;
    return (s * dinv);
}

#endif // OSKAR_INTERP_BILINEAR_H_
