/*
 * Copyright (c) 2013, The University of Oxford
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

#ifndef OSKAR_CONVERT_ECEF_TO_BASELINE_UVW_H_
#define OSKAR_CONVERT_ECEF_TO_BASELINE_UVW_H_

/**
 * @file oskar_convert_ecef_to_baseline_uvw.h
 */

#include <oskar_global.h>
#include <oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates the baseline (u,v,w) coordinates for the observation.
 *
 * @details
 * This function evaluates the baseline (u,v,w) coordinates from the beam phase
 * centre and station (x,y,z) coordinates in the telescope data structure,
 * and the supplied simulation time parameters.
 *
 * The output coordinates are for the whole observation, so the output arrays
 * must have dimension of (at least) num_baselines * num_vis_dumps.
 *
 * @param[out] uu           Output baseline u coordinates for whole observation.
 * @param[out] vv           Output baseline v coordinates for whole observation.
 * @param[out] ww           Output baseline w coordinates for whole observation.
 * @param[in]  num_stations The number of stations.
 * @param[in]  x            Station x coordinates (ECEF or related frame).
 * @param[in]  y            Station y coordinates (ECEF or related frame).
 * @param[in]  z            Station z coordinates (ECEF or related frame).
 * @param[in]  ra0_rad      The Right Ascension of the phase centre, in radians.
 * @param[in]  dec0_rad     The Declination of the phase centre, in radians.
 * @param[in]  num_dumps    The number of visibility dumps in the observation.
 * @param[in]  start_mjd_utc The start time of the observation.
 * @param[in]  dt_dump_days The time interval between dumps, in days.
 * @param[in,out]  work     Pointer to work buffer (>= 3 * num_stations).
 * @param[in,out]  status   Status return code.
 */
OSKAR_EXPORT
void oskar_convert_ecef_to_baseline_uvw(oskar_Mem* uu, oskar_Mem* vv, oskar_Mem* ww,
        int num_stations, const oskar_Mem* x, const oskar_Mem* y,
        const oskar_Mem* z, double ra0_rad, double dec0_rad, int num_dumps,
        double start_mjd_utc, double dt_dump_days, oskar_Mem* work,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_ECEF_TO_BASELINE_UVW_H_ */
