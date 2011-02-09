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

#ifndef SPHERICALPOSITIONS_H_
#define SPHERICALPOSITIONS_H_

#include <cmath>
#include "math/core/Geometry.h"

/**
 * @brief Class used for generating positions on a sphere.
 *
 * @details
 *
 */
template<typename T>
class SphericalPositions
{
public:
    /// Projection type.
    enum {
        PROJECTION_NONE,
        PROJECTION_SIN,
        PROJECTION_TAN,
        PROJECTION_ARC
    };

    /// Constructs a data structure for spherical position generation.
    SphericalPositions(const T centreLon, const T centreLat,
            const T sizeLon, const T sizeLat, const T sepLon, const T sepLat,
            const T rho = 0,
            const bool forceConstSep = true,
            const bool setCentreAfter = false,
            const bool forceCentrePoint = true,
            const bool forceToEdges = false,
            const int projectionType = PROJECTION_SIN);

    /// Generate the positions.
    unsigned generate(T* longitudes, T* latitudes) const;

private:
    // Parameters.
    T _lon0;
    T _lat0;
    T _sizeLon;
    T _sizeLat;
    T _sepLon;
    T _sepLat;
    T _rho;
    bool _forceConstSep;
    bool _setCentreAfter;
    bool _forceCentrePoint;
    bool _forceToEdges;
    int _projectionType;

    // Cached sines and cosines.
    T _sinLon0;
    T _cosLon0;
    T _sinLat0;
    T _cosLat0;
    T _sinRho;
    T _cosRho;
};

/**
 * @details
 * Pre-computes and stores the parameters required for generating
 * points on a sphere.
 *
 */
template<typename T>
SphericalPositions<T>::SphericalPositions(const T centreLon, const T centreLat,
        const T sizeLon, const T sizeLat, const T sepLon, const T sepLat,
        const T rho, const bool forceConstSep, const bool setCentreAfter,
        const bool forceCentrePoint, const bool forceToEdges,
        const int projectionType)
{
    _lon0 = centreLon;
    _lat0 = centreLat;
    _sizeLon = sizeLon;
    _sizeLat = sizeLat;
    _sepLon = sepLon;
    _sepLat = sepLat;
    _rho = rho;
    _forceConstSep = forceConstSep;
    _setCentreAfter = setCentreAfter;
    _forceCentrePoint = forceCentrePoint;
    _forceToEdges = forceToEdges;
    _projectionType = projectionType;

    // Create cache.
    _sinLon0 = sin(_lon0);
    _cosLon0 = cos(_lon0);
    _sinLat0 = sin(_lat0);
    _cosLat0 = cos(_lat0);
    _sinRho = sin(_rho);
    _cosRho = cos(_rho);
}

/**
 * @details
 * Generates the points using the parameters specified in the constructor
 * and fills the given arrays.
 *
 * @param[in,out] longitudes Pre-allocated array of longitude positions.
 * @param[in,out] latitudes  Pre-allocated array of latitude positions.
 */
template<typename T>
unsigned SphericalPositions<T>::generate(T* longitudes, T* latitudes) const
{
    // Catch divide-by-zero.
    if (_sepLon == 0 || _sepLat == 0) return 0;

    // Declare beam longitude and latitude.
    T lon = 0, lat = 0;
    unsigned nTotal = 0;

    // Set local centre, used only by these loops.
    T x0 = 0, y0 = 0;
    if (!_setCentreAfter && !_projectionType) {
        x0 = _lon0;
        y0 = _lat0;
    }

    // Set up y-axis (latitudes).
    T sepY = (!_projectionType) ? _sepLat : sin(_sepLat);
    int nY = 1 + floor(2 * _sizeLat / sepY); // Must be signed integer.
    if (_forceCentrePoint && nY % 2 == 0) nY--;
    T extentY = (_forceToEdges) ? (2 * _sizeLat) : (nY - 1) * sepY;
    T begY = y0 + extentY / 2;

    // Loop over y (latitude).
    for (int iY = 0; iY < nY; iY++) {
        T y = (nY == 1) ? begY : begY - iY * extentY / (nY - 1);

        // Compute cos(latitude) factor.
        T factor = 1;
        if (_forceConstSep && !_projectionType) {
            factor = cos(y);
            factor = (factor == 0) ? 1 : 1.0 / factor;
        }

        // Set up x-axis (longitudes).
        T sepX = (!_projectionType) ? (_sepLon * factor) : sin(_sepLon);
        int nX = 1 + floor(2 * _sizeLon / sepX); // Must be signed integer.
        if (_forceCentrePoint && nX % 2 == 0) nX--;
        T extentX = (_forceToEdges) ? (2 * _sizeLon) : (nX - 1) * sepX;
        T begX = x0 + extentX / 2;

        // Loop over x (longitude).
        for (int iX = 0; iX < nX; iX++) {
            T x = (nX == 1) ? begX : begX - iX * extentX / (nX - 1);

            // Get beam coordinates.
            if (!_projectionType) {
                lon = x;
                lat = y;
            } else {
                // Calculate L and M direction cosines.
                T L = x * _cosRho - y * _sinRho;
                T M = x * _sinRho + y * _cosRho;

                // Find which projection to use.
                if (_projectionType == PROJECTION_TAN) {
                    Geometry::tangentPlaneToSphericalGnomonic<T>(L, M,
                            lon, lat, _lon0, _sinLat0, _cosLat0);
                } else if (_projectionType == PROJECTION_SIN) {
                    if (!Geometry::tangentPlaneToSphericalOrthographic<T>(L,
                            M, lon, lat, _lon0, _sinLat0, _cosLat0)) continue;
                } else if (_projectionType == PROJECTION_ARC) {
                    Geometry::tangentPlaneToSphericalAzimuthalEquidistant<T>(
                            L, M, lon, lat, _lon0, _sinLat0, _cosLat0);
                }
            }

            // Store the beam coordinates if arrays exist.
            if (longitudes && latitudes) {
                if (_setCentreAfter && !_projectionType) {
                    // Set real centre, if needed.
                    T xt, yt, zt;
                    Geometry::cartesianFromHorizontal<T>(lon, lat, xt, yt, zt);
                    Geometry::rotateX<T>(_cosLat0, _sinLat0, xt, yt, zt,
                            xt, yt, zt);
                    Geometry::rotateZ<T>(_cosLon0, -_sinLon0, xt, yt, zt,
                            xt, yt, zt);
                    Geometry::cartesianToHorizontal<T>(xt, yt, zt, lon, lat);
                }
                longitudes[nTotal] = lon;
                latitudes[nTotal] = lat;
            }
            nTotal++;
        }
    }

    return nTotal;
}

#endif /* SPHERICALPOSITIONS_H_ */
