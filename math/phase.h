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

#ifndef OSKAR_PHASE_H_
#define OSKAR_PHASE_H_

/**
 * @file phase.h
 */

/**
 * @brief
 * Inline function macro used to compute the 2D geometric phase
 * for the horizontal (azimuth/elevation) coordinate system.
 *
 * - Antenna positions are specified in the horizontal xy-plane, where the
 *   x-axis points to geographic East, y to geographic North, and z to the
 *   zenith.
 * - Azimuth is the angle measured from the y-axis (North) towards the
 *   x-axis (East).
 * - Elevation is the angle measured from the xy-plane towards the z-axis,
 *   which points to the zenith.
 */
#define GEOMETRIC_PHASE_2D_HORIZONTAL(x, y, cosEl, sinAz, cosAz, k) \
        (-k * cosEl * (x * sinAz + y * cosAz))

/**
 * @brief
 * Inline function macro used to compute the 2D geometric phase
 * for the horizontal (azimuth/elevation) coordinate system.
 *
 * Positions are assumed to be already multiplied by the wavenumber.
 *
 * - Antenna positions are specified in the horizontal xy-plane, where the
 *   x-axis points to geographic East, y to geographic North, and z to the
 *   zenith.
 * - Azimuth is the angle measured from the y-axis (North) towards the
 *   x-axis (East).
 * - Elevation is the angle measured from the xy-plane towards the z-axis,
 *   which points to the zenith.
 */
#define GEOMETRIC_PHASE_2D_HORIZONTAL_K(x, y, cosEl, sinAz, cosAz) \
        (-cosEl * (x * sinAz + y * cosAz))

/**
 * @brief
 * Inline function macro used to compute the 2D geometric phase
 * for the spherical (phi/theta) coordinate system.
 *
 * - Antenna positions are specified in the horizontal xy-plane, where the
 *   x-axis points to geographic East, y to geographic North, and z to the
 *   zenith.
 * - Phi is the angle measured from the x-axis towards the y-axis.
 * - Theta is the angle measured from the z-axis (the zenith angle).
 */
#define GEOMETRIC_PHASE_2D_SPHERICAL(x, y, sinTheta, cosPhi, sinPhi, k) \
        (-k * sinTheta * (x * cosPhi + y * sinPhi))

/**
 * @brief
 * Inline function macro used to compute the 2D geometric phase
 * for the spherical (phi/theta) coordinate system.
 *
 * Positions are assumed to be already multiplied by the wavenumber.
 *
 * - Antenna positions are specified in the horizontal xy-plane, where the
 *   x-axis points to geographic East, y to geographic North, and z to the
 *   zenith.
 * - Phi is the angle measured from the x-axis towards the y-axis.
 * - Theta is the angle measured from the z-axis (the zenith angle).
 */
#define GEOMETRIC_PHASE_2D_SPHERICAL_K(x, y, sinTheta, cosPhi, sinPhi) \
        (-sinTheta * (x * cosPhi + y * sinPhi))

/**
 * @brief
 * Inline function macro used to compute the 3D geometric phase
 * for the horizontal (azimuth/elevation) coordinate system.
 *
 * - Antenna positions are specified in 3D space, where the x-axis points to
 *   geographic East, y to geographic North, and z to the zenith.
 * - Azimuth is the angle measured from the y-axis (North) towards the
 *   x-axis (East).
 * - Elevation is the angle measured from the xy-plane towards the z-axis,
 *   which points to the zenith.
 */
#define GEOMETRIC_PHASE_3D_HORIZONTAL(x, y, z, sinEl, cosEl, sinAz, cosAz, k) \
        (-k * (cosEl * (x * sinAz + y * cosAz) + z * sinEl))

/**
 * @brief
 * Inline function macro used to compute the 3D geometric phase
 * for the horizontal (azimuth/elevation) coordinate system.
 *
 * Positions are assumed to be already multiplied by the wavenumber.
 *
 * - Antenna positions are specified in 3D space, where the x-axis points to
 *   geographic East, y to geographic North, and z to the zenith.
 * - Azimuth is the angle measured from the y-axis (North) towards the
 *   x-axis (East).
 * - Elevation is the angle measured from the xy-plane towards the z-axis,
 *   which points to the zenith.
 */
#define GEOMETRIC_PHASE_3D_HORIZONTAL_K(x, y, z, sinEl, cosEl, sinAz, cosAz) \
        (-(cosEl * (x * sinAz + y * cosAz) + z * sinEl))

/**
 * @brief
 * Inline function macro used to compute the 3D geometric phase
 * for the local equatorial (hour angle/declination) coordinate system.
 *
 * - Antenna positions are specified in 3D space, where the x-axis points to
 *   the local meridian, y towards the East, and z along the Earth's axis of
 *   rotation towards the North Celestial Pole.
 * - The Hour Angle is the angle measured towards the West from the observer's
 *   meridian, but in the equatorial system.
 * - The Declination is the angle measured from the Earth's equator towards the
 *   North Celestial Pole.
 */
#define GEOMETRIC_PHASE_3D_LOCAL_EQUATORIAL(x, y, z, \
        sinDec, cosDec, sinHA, cosHA, k) \
        (-k * (cosDec * (x * cosHA - y * sinHA) + z * sinDec))

/**
 * @brief
 * Inline function macro used to compute the 3D geometric phase
 * for the local equatorial (hour angle/declination) coordinate system.
 *
 * Positions are assumed to be already multiplied by the wavenumber.
 *
 * - Antenna positions are specified in 3D space, where the x-axis points to
 *   the local meridian, y towards the East, and z along the Earth's axis of
 *   rotation towards the North Celestial Pole.
 * - The Hour Angle is the angle measured towards the West from the observer's
 *   meridian, but in the equatorial system.
 * - The Declination is the angle measured from the Earth's equator towards the
 *   North Celestial Pole.
 */
#define GEOMETRIC_PHASE_3D_LOCAL_EQUATORIAL_K(x, y, z, \
        sinDec, cosDec, sinHA, cosHA) \
        (-(cosDec * (x * cosHA - y * sinHA) + z * sinDec))

#endif // OSKAR_PHASE_H_
