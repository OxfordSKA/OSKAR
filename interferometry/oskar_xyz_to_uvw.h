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

#ifndef OSKAR_INTERFEROMETRY_XYZ2UVW_H_
#define OSKAR_INTERFEROMETRY_XYZ2UVW_H_

/**
 * @file oskar_xyz2uvw.h
 */

#include "oskar_windows.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Transforms station (x,y,z) coordinates to (u,v,w) coordinates
 * (single precision).
 *
 * @details
 * Given the hour angle and declination of the phase tracking centre, this
 * function transforms the station (x,y,z) coordinates to (u,v,w) coordinates.
 *
 * @param[in]  na   The number of antennas.
 * @param[in]  ax   The station x-positions (length na).
 * @param[in]  ay   The station y-positions (length na).
 * @param[in]  az   The station z-positions (length na).
 * @param[in]  ha0  The hour angle of the phase tracking centre in radians.
 * @param[in]  dec0 The declination of the phase tracking centre in radians.
 * @param[out] u    The station u-positions (length na).
 * @param[out] v    The station v-positions (length na).
 * @param[out] w    The station w-positions (length na).
 */
DllExport
void oskar_xyz_to_uvw_f(int na, const float* x,
        const float* y, const float* z, double ha0, double dec0, float* u,
        float* v, float* w);

/**
 * @brief
 * Transforms station (x,y,z) coordinates to (u,v,w) coordinates
 * (double precision).
 *
 * @details
 * Given the hour angle and declination of the phase tracking centre, this
 * function transforms the station (x,y,z) coordinates to (u,v,w) coordinates.
 *
 * @param[in]  na   The number of antennas.
 * @param[in]  ax   The station x-positions (length na).
 * @param[in]  ay   The station y-positions (length na).
 * @param[in]  az   The station z-positions (length na).
 * @param[in]  ha0  The hour angle of the phase tracking centre in radians.
 * @param[in]  dec0 The declination of the phase tracking centre in radians.
 * @param[out] u    The station u-positions (length na).
 * @param[out] v    The station v-positions (length na).
 * @param[out] w    The station w-positions (length na).
 */
DllExport
void oskar_xyz_to_uvw_d(int na, const double* x,
        const double* y, const double* z, double ha0, double dec0, double* u,
        double* v, double* w);

#ifdef __cplusplus
}
#endif

#endif // OSKAR_INTERFEROMETRY_XYZ2UVW_H_
