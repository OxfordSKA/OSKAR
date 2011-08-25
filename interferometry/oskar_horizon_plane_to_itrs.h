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

#ifndef OSKAR_HORIZON_PLANE_TO_ITRS_H_
#define OSKAR_HORIZON_PLANE_TO_ITRS_H_

/**
 * @file oskar_horizon_plane_to_itrs.h
 */

#include "oskar_windows.h"
#include "interferometry/oskar_TelescopeModel.h"

#ifdef __cplusplus
extern "C" {
#endif


/**
 * @brief
 * Convert antenna coordinates to ITRS system.
 *
 * @details
 *
 * Transform the station positions to the local equatorial system.
 * x-coordinates is towards the local Meridian, y-coordinates to the East,
 * and z-coordinates to the North Pole.
 *
 * see Thompson, Swenson & Moran (ch4, p86)
 *
 * @param[in]  num_antennas  Number of antennas/ stations.
 * @param[in]  x_horizon     Vector of horizontal station x-positions, in metres.
 * @param[in]  y_horizon     Vector of horizontal station y-positions, in metres.
 * @param[in]  latitude      Telescope latitude, in radians.
 * @param[out] x             Vector of ITRS x-positions, in metres.
 * @param[out] y             Vector of ITRS x-positions, in metres.
 * @param[out] z             Vector of ITRS x-positions, in metres.
 */
DllExport
void oskar_horizon_plane_to_itrs(const unsigned num_antennas,
        const double * x_horizon, const double * y_horizon,
        const double latitude, double * x, double * y, double * z);


#ifdef __cplusplus
}
#endif


#endif // OSKAR_HORIZON_PLANE_TO_ITRS_H_
