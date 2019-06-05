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

#ifndef OSKAR_CONVERT_ECEF_TO_UVW_H_
#define OSKAR_CONVERT_ECEF_TO_UVW_H_

/**
 * @file oskar_convert_ecef_to_uvw.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates (u,v,w) coordinates using observation parameters.
 *
 * @details
 * This function evaluates the station and, optionally, baseline (u,v,w)
 * coordinates using the supplied ECEF station (x,y,z) positions,
 * phase centre and simulation time parameters.
 *
 * The baseline coordinates are not computed if \p uu, \p vv or \p ww are NULL.
 *
 * @param[in]  num_stations     The number of stations.
 * @param[in]  x                Station x coordinates (ECEF or related frame).
 * @param[in]  y                Station y coordinates (ECEF or related frame).
 * @param[in]  z                Station z coordinates (ECEF or related frame).
 * @param[in]  ra0_rad          Right Ascension of the phase centre, in radians.
 * @param[in]  dec0_rad         Declination of the phase centre, in radians.
 * @param[in]  num_times        Number of time steps to loop over.
 * @param[in]  time_ref_mjd_utc Start time of the observation.
 * @param[in]  time_inc_days    Time interval, in days.
 * @param[in]  start_time_index Time index for the start of the block.
 * @param[in]  ignore_w_components If true, set all output w coordinates to 0.
 * @param[out] u                Output station u coordinates.
 * @param[out] v                Output station v coordinates.
 * @param[out] w                Output station w coordinates.
 * @param[out] uu               Optional output baseline u coordinates.
 * @param[out] vv               Optional output baseline v coordinates.
 * @param[out] ww               Optional output baseline w coordinates.
 * @param[in,out]  status       Status return code.
 */
OSKAR_EXPORT
void oskar_convert_ecef_to_uvw(int num_stations,
        const oskar_Mem* x, const oskar_Mem* y, const oskar_Mem* z,
        double ra0_rad, double dec0_rad, int num_times,
        double time_ref_mjd_utc, double time_inc_days, int start_time_index,
        int ignore_w_components, oskar_Mem* u, oskar_Mem* v, oskar_Mem* w,
        oskar_Mem* uu, oskar_Mem* vv, oskar_Mem* ww, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
