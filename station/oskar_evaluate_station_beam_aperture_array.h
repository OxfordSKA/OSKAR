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


#ifndef OSKAR_EVALUATE_STATION_BEAM_AA_H_
#define OSKAR_EVALUATE_STATION_BEAM_AA_H_

/**
 * @file oskar_evaluate_station_beam_AA.h
 */

#include "oskar_global.h"
#include "station/oskar_StationModel.h"
#include "station/oskar_WorkStationBeam.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_CurandState.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Evaluates the station beam (E-Jones) for an aperture array (AA)
 * station.
 *
 * @details
 * The beam is evaluated at points defined by the horizontal Cartesian
 * coordinates x,y,z.
 *
 * TODO better description of what this does
 *
 * TODO better description of how to call this function... examples..?
 *
 * Notes:
 * - work buffer .... TODO what parts of this is used how to set up
 * - curand_states .. TODO how to set up this buffer
 *
 * @param[out]    beam          Station beam returned
 * @param[in]     station       Station model structure
 * @param[in]     beam_x        Beam direction in horizontal coordinates.
 * @param[in]     beam_y        Beam direction in horizontal coordinates.
 * @param[in]     beam_z        Beam direction in horizontal coordinates.
 * @param[in]     num_points    Number of coordinates at which to evaluate
 *                              the beam pattern
 * @param[in]     x             Array of horizontal x coordinates at which to
 *                              evaluate the beam.
 * @param[in]     y             Array of horizontal y coordinates at which to
 *                              evaluate the beam.
 * @param[in]     z             Array of horizontal z coordinates at which to
 *                              evaluate the beam.
 * @param[in]     work          Structure containing temporary work buffers
 * @param[in]     curand_states Array of CUDA random number states used
 *                              for various station model errors.
 * @param[in/out] status        OSKAR status code.
 */
OSKAR_EXPORT
void oskar_evaluate_station_beam_aperture_array(oskar_Mem* beam,
        const oskar_StationModel* station, double beam_x, double beam_y,
        double beam_z, int num_points, const oskar_Mem* x, const oskar_Mem* y,
        const oskar_Mem* z, oskar_WorkStationBeam* work,
        oskar_CurandState* curand_states, int* status);


#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_STATION_BEAM_AA_H_ */
