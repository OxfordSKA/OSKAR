/*
 * Copyright (c) 2013-2019, The University of Oxford
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

#ifndef OSKAR_CONVERT_ENU_DIRECTIONS_TO_RELATIVE_DIRECTIONS_H_
#define OSKAR_CONVERT_ENU_DIRECTIONS_TO_RELATIVE_DIRECTIONS_H_

/**
 * @file oskar_convert_enu_directions_to_relative_directions.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Converts from horizon ENU direction cosines to relative direction cosines.
 *
 * @details
 * This function transforms the given \f$(l, m, n)\f$ directions in the
 * equatorial frame to \f$(x, y, z)\f$ directions in the horizontal frame.
 *
 * It is equivalent to the product of matrix transformations as follows:
 *
   \f[
    \begin{bmatrix}
    l \\
    m \\
    n
    \end{bmatrix}

        = R_x(\delta_0) \cdot R_y(H_0) \cdot R_x(-\phi) \cdot

    \begin{bmatrix}
    x \\
    y \\
    z
    \end{bmatrix}
   \f]
 *
 * Here, \f$ R_x \f$ and \f$ R_y \f$ correspond to rotations around
 * the \f$x\f$- and \f$y\f$-axes, respectively.
 * The angles \f$ \phi \f$, \f$ H_0 \f$ and \f$ \delta_0 \f$ correspond to
 * the observer's geodetic latitude, the hour angle and the declination of
 * the phase centre.
 *
 * @param[in]  num_points Number of points to convert.
 * @param[in]  x          ENU direction cosines (East).
 * @param[in]  y          ENU direction cosines (North).
 * @param[in]  z          ENU direction cosines (up).
 * @param[in]  ha0        Hour angle of the origin of the relative directions,
 *                        in radians.
 * @param[in]  dec0       Declination of the origin of the relative directions,
 *                        in radians.
 * @param[in]  lat        Latitude of the ENU coordinate frame, in radians.
 * @param[out] l          Relative direction cosines.
 * @param[out] m          Relative direction cosines.
 * @param[out] n          Relative direction cosines.
 */
OSKAR_EXPORT
void oskar_convert_enu_directions_to_relative_directions(int offset_in,
        int num_points, const oskar_Mem* x, const oskar_Mem* y,
        const oskar_Mem* z, double ha0, double dec0, double lat,
        int offset_out, oskar_Mem* l, oskar_Mem* m, oskar_Mem* n, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_ENU_DIRECTIONS_TO_RELATIVE_DIRECTIONS_H_ */
