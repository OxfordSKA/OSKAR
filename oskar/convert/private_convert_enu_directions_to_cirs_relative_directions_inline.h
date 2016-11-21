/*
 * Copyright (c) 2014, The University of Oxford
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

#ifndef OSKAR_PRIVATE_CONVERT_ENU_DIRECTIONS_TO_CIRS_RELATIVE_DIRECTIONS_INLINE_H_
#define OSKAR_PRIVATE_CONVERT_ENU_DIRECTIONS_TO_CIRS_RELATIVE_DIRECTIONS_INLINE_H_

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
OSKAR_INLINE
void oskar_convert_enu_directions_to_cirs_relative_directions_inline_f(
        float x, float y, float z, const float sin_lat,
        const float cos_lat, const float sin_ha0, const float cos_ha0,
        const float sin_dec0, const float cos_dec0, const float local_pm_x,
        const float local_pm_y, const float diurnal_aberration,
        float* l, float* m, float* n)
{
    float f, w, x2, y2, z2, t1, t2;

    /* ENU directions to Cartesian -HA,Dec. */
    /* This is the first (latitude) stage of the original transformation:
     *   rotate by -phi about x.
     * INTERNAL: Axes are then permuted so that x -> Y, y -> Z and z -> X.
     * X towards local meridian, Z towards NCP, Y towards local east. */
    y2 = x;
    z2 = y * cos_lat + z * sin_lat;
    x2 = -y * sin_lat + z * cos_lat;

    /* Diurnal aberration. */
    f = 1.0f + diurnal_aberration * y2;
    x = f * x2;
    y = f * (y2 - diurnal_aberration);
    z = f * z2;

    /* Polar motion, permuting axes back to ENU directions. */
    w = local_pm_x * x - local_pm_y * y + z;
    z2 = x - local_pm_x * w;
    x2 = y + local_pm_y * w;
    y2 = w - (local_pm_x * local_pm_x + local_pm_y * local_pm_y) * z;

    /* ENU directions to CIRS relative directions. */
    /* This is the final two stages of the original transformation:
     *   rotate by ha0 around y, then by dec0 around x. */
    t1 = x2 * sin_ha0;
    t2 = z2 * cos_ha0;
    *l =  x2 * cos_ha0 + z2 * sin_ha0;
    *m =  t1 * sin_dec0 + y2 * cos_dec0 - t2 * sin_dec0;
    *n = -t1 * cos_dec0 + y2 * sin_dec0 + t2 * cos_dec0;
}


/* Double precision. */
OSKAR_INLINE
void oskar_convert_enu_directions_to_cirs_relative_directions_inline_d(
        double x, double y, double z, const double sin_lat,
        const double cos_lat, const double sin_ha0, const double cos_ha0,
        const double sin_dec0, const double cos_dec0, const double local_pm_x,
        const double local_pm_y, const double diurnal_aberration,
        double* l, double* m, double* n)
{
    double f, w, x2, y2, z2, t1, t2;

    /* ENU directions to Cartesian -HA,Dec. */
    /* This is the first (latitude) stage of the original transformation:
     *   rotate by -phi about x.
     * INTERNAL: Axes are then permuted so that x -> Y, y -> Z and z -> X.
     * X towards local meridian, Z towards NCP, Y towards local east. */
    y2 = x;
    z2 = y * cos_lat + z * sin_lat;
    x2 = -y * sin_lat + z * cos_lat;

    /* Diurnal aberration. */
    f = 1.0 + diurnal_aberration * y2;
    x = f * x2;
    y = f * (y2 - diurnal_aberration);
    z = f * z2;

    /* Polar motion, permuting axes back to ENU directions. */
    w = local_pm_x * x - local_pm_y * y + z;
    z2 = x - local_pm_x * w;
    x2 = y + local_pm_y * w;
    y2 = w - (local_pm_x * local_pm_x + local_pm_y * local_pm_y) * z;

    /* ENU directions to CIRS relative directions. */
    /* This is the final two stages of the original transformation:
     *   rotate by ha0 around y, then by dec0 around x. */
    t1 = x2 * sin_ha0;
    t2 = z2 * cos_ha0;
    *l =  x2 * cos_ha0 + z2 * sin_ha0;
    *m =  t1 * sin_dec0 + y2 * cos_dec0 - t2 * sin_dec0;
    *n = -t1 * cos_dec0 + y2 * sin_dec0 + t2 * cos_dec0;
}

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_PRIVATE_CONVERT_ENU_DIRECTIONS_TO_CIRS_RELATIVE_DIRECTIONS_INLINE_H_ */
