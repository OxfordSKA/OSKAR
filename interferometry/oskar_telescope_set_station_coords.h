/*
 * Copyright (c) 2011-2014, The University of Oxford
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

#ifndef OSKAR_TELESCOPE_SET_STATION_COORDS_H_
#define OSKAR_TELESCOPE_SET_STATION_COORDS_H_

/**
 * @file oskar_telescope_set_station_coords.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Sets the coordinates of a station in the telescope model.
 *
 * @details
 * This function sets the coordinates of the specified station in the telescope
 * model, transferring data to the GPU if necessary.
 *
 * All coordinates must be in metres.
 *
 * @param[in] dst           Telescope model structure to modify.
 * @param[in] index         Station array index to set.
 * @param[in] x_offset_ecef Station x position (ECEF).
 * @param[in] y_offset_ecef Station y position (ECEF).
 * @param[in] z_offset_ecef Station z position (ECEF).
 * @param[in] x_enu         Station x position (horizon).
 * @param[in] y_enu         Station y position (horizon).
 * @param[in] z_enu         Station z position (horizon).
 * @param[in,out] status    Status return code.
 */
OSKAR_EXPORT
void oskar_telescope_set_station_coords(oskar_Telescope* dst, int index,
        double x_offset_ecef, double y_offset_ecef, double z_offset_ecef,
        double x_enu, double y_enu, double z_enu, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_TELESCOPE_SET_STATION_COORDS_H_ */
