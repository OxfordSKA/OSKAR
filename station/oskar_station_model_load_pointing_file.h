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

#ifndef OSKAR_STATION_MODEL_LOAD_POINTING_FILE_H_
#define OSKAR_STATION_MODEL_LOAD_POINTING_FILE_H_

/**
 * @file oskar_station_model_load_pointing_file.h
 */

#include "oskar_global.h"
#include "station/oskar_StationModel.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Loads a station pointing file, which can be used to specify the direction
 * of the beamformed phase centre for every (sub-)station in the array.
 *
 * @details
 * This function loads a text file specifying the direction of the required
 * beam for all detectors in the station model.
 *
 * The text file has five columns, which specify:
 * - The depth of the beamforming level (0 is top-level; higher numbers go
 *   deeper).
 * - The detector ID at that depth.
 * - The longitude of the beam in degrees.
 * - The latitude of the beam in degrees.
 * - The coordinate system used for the beam specification.
 *   This is a string which may be either AZEL or RADEC to specify
 *   horizontal or equatorial coordinates.
 *
 * Wild-cards (an asterisk, *) may be used in the first two columns to allow
 * the same direction for all depths or all stations at a given depth.
 *
 * For example, a file may contain the following two lines to specify different
 * phase centres for beams formed at the tile and station levels:
 *
 * 0, *, 45.0, 60.0, RADEC
 * 1, *, 60.0, 75.0, AZEL
 *
 * @param[in,out] station    Station model structure to modify.
 * @param[in] filename       File name path to a pointing file.
 * @param[in,out] status     Status return code.
 */
OSKAR_EXPORT
void oskar_station_model_load_pointing_file(oskar_StationModel* station,
        const char* filename, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_STATION_MODEL_LOAD_POINTING_FILE_H_ */
