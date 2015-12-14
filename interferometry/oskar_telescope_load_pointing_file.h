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

#ifndef OSKAR_TELESCOPE_LOAD_POINTING_FILE_H_
#define OSKAR_TELESCOPE_LOAD_POINTING_FILE_H_

/**
 * @file oskar_telescope_load_pointing_file.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Loads a telescope model pointing file, which can be used to specify the
 * direction of the beamformed phase centre for every (sub-)station in the
 * array.
 *
 * @details
 * This function loads a text file specifying the direction of the required
 * beam for any or all stations in the telescope model.
 *
 * The text file has multiple columns, which specify the address of the
 * station(s) in the hierarchy (via multiple indices) and the beam direction
 * to set for the station(s). The columns are:
 *
 * - The index of the top-level station.
 * - The index of the station at the next level down, if required.
 * - ... (and so on for further sub-stations).
 * - The coordinate system used for the beam specification.
 *   This is a string which may be either AZEL or RADEC to specify
 *   horizontal or equatorial coordinates.
 * - The longitude of the beam in degrees.
 * - The latitude of the beam in degrees.
 *
 * Wild-cards (an asterisk, *) may be used in the index columns to allow
 * the same direction for all stations at a given depth.
 *
 * An entry in the file will set the beam direction for the station(s) at the
 * last specified index, and recursively for all child stations.
 *
 * Note that the order in which lines appear in the file is important.
 * Entries that appear later override those that appear earlier.
 *
 * For example, a file may contain the following lines to specify different
 * phase centres for beams formed at the tile and station levels:
 *
   @verbatim
   *   RADEC 45.0 60.0 # All stations (and children) track (RA, Dec) = (45, 60).
   3   RADEC 45.1 59.9 # Station 3 (and children) tracks (RA, Dec) = (45.1, 59.9).
   * * AZEL  60.0 75.0 # All tiles in all stations have fixed beams.
   0 * AZEL  60.1 75.0 # All tiles in station 0 are offset from the rest.
   2 6 AZEL   0.0 90.0 # Tile 6 in station 2 is pointing at the zenith.
   @endverbatim
 *
 * @param[in,out] telescope  Telescope model structure to modify.
 * @param[in] filename       File name path to a pointing file.
 * @param[in,out] status     Status return code.
 */
OSKAR_EXPORT
void oskar_telescope_load_pointing_file(oskar_Telescope* telescope,
        const char* filename, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_TELESCOPE_LOAD_POINTING_FILE_H_ */
