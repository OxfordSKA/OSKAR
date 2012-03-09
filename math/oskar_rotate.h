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


#ifndef OSKAR_ROTATE_H_
#define OSKAR_ROTATE_H_

/**
 * @file oskar_rotate.h
 */

#include "oskar_global.h"
#include "utility/oskar_Mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief rotate a set of points about the x axis.
 *
 * @param n         Number of points to rotate.
 * @param x         x coordinates.
 * @param y         y coordinates.
 * @param z         z coordinates.
 * @param angle     rotation angle, in radians.
 *
 * @return An error code.
 */
OSKAR_EXPORT
int oskar_rotate_x(int n, oskar_Mem* x, oskar_Mem* y, oskar_Mem* z,
        double angle);


/**
 * @brief rotate a set of points about the y axis.
 *
 * @param n         Number of points to rotate.
 * @param x         x coordinates.
 * @param y         y coordinates.
 * @param z         z coordinates.
 * @param angle     rotation angle, in radians.
 *
 * @return An error code.
 */
OSKAR_EXPORT
int oskar_rotate_y(int n, oskar_Mem* x, oskar_Mem* y, oskar_Mem* z,
        double angle);


/**
 * @brief rotate a set of points about the z axis.
 *
 * @param n         Number of points to rotate.
 * @param x         x coordinates.
 * @param y         y coordinates.
 * @param z         z coordinates.
 * @param angle     rotation angle, in radians.
 *
 * @return An error code.
 */
OSKAR_EXPORT
int oskar_rotate_z(int n, oskar_Mem* x, oskar_Mem* y, oskar_Mem* z,
        double angle);

/**
 * @brief Rotate on the surface of a sphere by the specified longitude
 * and latitude.
 *
 * @param n         Number of points to rotate.
 * @param x         x coordinates.
 * @param y         y coordinates.
 * @param z         z coordinates.
 * @param lon       East longitude rotation angle, in radians.
 * @param lat       North latitude rotation angle, in radians.
 *
 * @return An error code.
 */
OSKAR_EXPORT
int oskar_rotate_sph(int n, oskar_Mem* x, oskar_Mem* y, oskar_Mem* z,
        double lon, double lat);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_ROTATE_H_ */
