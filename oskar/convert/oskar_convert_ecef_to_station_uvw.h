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

#ifndef OSKAR_CONVERT_ECEF_TO_STATION_UVW_H_
#define OSKAR_CONVERT_ECEF_TO_STATION_UVW_H_

/**
 * @file oskar_convert_ecef_to_station_uvw.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates the station (u,v,w) coordinates.
 *
 * @details
 * This function evaluates the station (u,v,w) coordinates using the
 * station (x,y,z) coordinates, the supplied phase tracking centre, and
 * the Greenwich Apparent Sidereal Time.
 *
 * @param[in]  num_stations The size of the station coordinate arrays.
 * @param[in]  x            Input station x coordinates (ECEF or related frame).
 * @param[in]  y            Input station y coordinates (ECEF or related frame).
 * @param[in]  z            Input station z coordinates (ECEF or related frame).
 * @param[in]  ra0_rad      The Right Ascension of the phase centre, in radians.
 * @param[in]  dec0_rad     The Declination of the phase centre, in radians.
 * @param[in]  gast         The Greenwich Apparent Sidereal Time, in radians.
 * @param[in]  ignore_w_components If true, set all output w coordinates to 0.
 * @param[in]  offset_out   Start offset into output arrays.
 * @param[out] u            Output station u coordinates.
 * @param[out] v            Output station v coordinates.
 * @param[out] w            Output station w coordinates.
 * @param[in,out] status    Status return code.
 */
OSKAR_EXPORT
void oskar_convert_ecef_to_station_uvw(int num_stations,
        const oskar_Mem* x, const oskar_Mem* y, const oskar_Mem* z,
        double ra0_rad, double dec0_rad, double gast, int ignore_w_components,
        int offset_out, oskar_Mem* u, oskar_Mem* v, oskar_Mem* w, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_ECEF_TO_STATION_UVW_H_ */
