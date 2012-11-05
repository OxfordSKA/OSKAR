/*
 * Copyright (c) 2012, The University of Oxford
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

#ifndef OSKAR_TELESCOPE_MODEL_SET_STATION_COORDS_H_
#define OSKAR_TELESCOPE_MODEL_SET_STATION_COORDS_H_

/**
 * @file oskar_telescope_model_set_station_coords.h
 */

#include "oskar_global.h"
#include "interferometry/oskar_TelescopeModel.h"

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
 * @param[in] dst        Telescope model structure to copy into.
 * @param[in] index      Station array index to set.
 * @param[in] x          Station x position (ECEF).
 * @param[in] y          Station y position (ECEF).
 * @param[in] z          Station z position (ECEF).
 * @param[in] x_hor      Station x position (horizon plane).
 * @param[in] y_hor      Station y position (horizon plane).
 * @param[in] z_hor      Station z position (horizon plane).
 * @param[in,out] status Status return code.
 */
OSKAR_EXPORT
void oskar_telescope_model_set_station_coords(oskar_TelescopeModel* dst,
        int index, double x, double y, double z,
        double x_hor, double y_hor, double z_hor, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_TELESCOPE_MODEL_SET_STATION_COORDS_H_ */
