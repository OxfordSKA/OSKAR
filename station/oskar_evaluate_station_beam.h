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

#ifndef OSKAR_EVALUATE_STATION_BEAM_H_
#define OSKAR_EVALUATE_STATION_BEAM_H_

/**
 * @file oskar_evaluate_station_beam.h
 */

#include "oskar_global.h"
#include "station/oskar_StationModel.h"
#include "station/oskar_WorkStationBeam.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_Device_curand_state.h"

#ifdef __cplusplus
extern "C" {
#endif

enum OSKAR_STATION_BEAM_COORD_TYPE { HORIZONTAL_XYZ, PHASE_CENTRE_XYZ };
typedef enum OSKAR_STATION_BEAM_COORD_TYPE oskar_station_beam_coord_type;

/**
 * @brief Evaluates the value of a station beam at a number of discrete
 * positions for the given station and beam direction. This is equivalent
 * to the E-Jones matrices for a given station.
 *
 * @details
 * The station beam amplitudes are evaluated using a DFT on the GPU, so
 * all memory passed to and returned from this function must be allocated
 * on the device.
 *
 * The detailed description of processing performed by this function will
 * depend on the presence of element pattern and hierarchical layout information
 * within the station structure.
 *
 * Note:
 * - Station x,y,z coordinates used by this function are assumed to be in
 * radians (i.e. pre-multiplied by the wavenumber).
 * - The \p weights buffer must be allocated on the GPU of complex type
 * matching the same floating point precision as the rest of the memory
 * passed to the function.
 * - Horizontal n (\p hor_n) coordinates are used to remove sources below the
 * horizon (i.e. where n < 0).
 *
 * @param[out] EG            Array of station complex beam amplitudes returned.
 * @param[in]  station       Station model structure.
 * @param[in]  l_beam        Beam phase centre horizontal l (component along x).
 * @param[in]  m_beam        Beam phase centre horizontal m (component along y).
 * @param[in]  n_beam        Beam phase centre horizontal n (component along z).
 * @param[in]  num_points    Number of points at which to evaluate beam.
 * @param[in]  l             Array of horizontal l directions for which the beam
 *                           should be evaluated (component along x).
 * @param[in]  m             Array of horizontal m directions for which the beam
 *                           should be evaluated (component along y).
 * @param[in]  n             Array of horizontal m directions for which the beam
 *                           should be evaluated (component along z).
 * @param[in]  work          Pointer to a work buffer.
 * @param[in]  curand_states Structure holding a set of curand states.
 *
 * @return An error code.
 */
OSKAR_EXPORT
void oskar_evaluate_station_beam(oskar_Mem* beam,
        const oskar_StationModel* station, double beam_x, double beam_y,
        double beam_z, int num_points, oskar_station_beam_coord_type type,
        const oskar_Mem* x, const oskar_Mem* y, const oskar_Mem* z,
        const oskar_Mem* horizon_mask, oskar_WorkStationBeam* work,
        oskar_Device_curand_state* curand_states, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_STATION_BEAM_H_ */
