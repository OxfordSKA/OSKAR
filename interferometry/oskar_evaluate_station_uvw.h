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

#ifndef OSKAR_EVALUATE_STATION_UVW_H_
#define OSKAR_EVALUATE_STATION_UVW_H_

/**
 * @file oskar_evaluate_station_uvw.h
 */

#include "oskar_global.h"
#include "interferometry/oskar_TelescopeModel.h"
#include "utility/oskar_Mem.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates the station (u,v,w) coordinates.
 *
 * @details
 * This function evaluates the station (u,v,w) coordinates from the beam phase
 * centre and station (x,y,z) coordinates in the telescope data structure,
 * and the supplied Greenwich Apparent Sidereal Time.
 *
 * Note:
 * The units of u,v,w returned by the function depend on the units of station
 * coordinates in the telescope model. For oskar_interferometer() this is
 * assumed to be in radians at the current simulation frequency. This is achieved
 * by scaling coordinates in a station model by the wavenumber cooresponding
 * to the simulation frequency..
 *
 * @param[out] u         Station u coordinates, in telescope->coord_units units.
 * @param[out] v         Station v coordinates, in telescope->coord_units units.
 * @param[out] w         Station w coordinates, in telescope->coord_units units.
 * @param[in] telescope  Input telescope model.
 * @param[in] gast       The Greenwich Apparent Sidereal Time, in radians.
 */
OSKAR_EXPORT
int oskar_evaluate_station_uvw(oskar_Mem* u, oskar_Mem* v, oskar_Mem* w,
        const oskar_TelescopeModel* telescope, double gast);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_STATION_UVW_H_ */
