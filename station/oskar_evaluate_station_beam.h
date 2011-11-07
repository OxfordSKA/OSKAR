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

#ifndef OSKAR_EVALUATE_STATION_BEAM_H_
#define OSKAR_EVALUATE_STATION_BEAM_H_

/**
 * @file oskar_evaluate_station_beam.h
 */

#include "oskar_global.h"
#include "utility/oskar_Mem.h"
#include "station/oskar_StationModel.h"
#include "math/oskar_Jones.h"
#include "sky/oskar_SkyModel.h" // no longer needed when old functions are removed.

#ifdef __cplusplus
extern "C" {
#endif


/**
 * @brief Evaluates the value of a station beam at a number of discrete
 * positions for the given station and beam direction. This is equivalent
 * to the Jones-E matrices for a given station.
 *
 * @details
 * The station beam values are evaluated using a DFT on the GPU and as such
 * all memory, passed to and return from this function, must be allocated
 * in device memory prior to calling this function.
 *
 * The detailed description of processing performed by this function will
 * depend on the presence of element pattern and hierarchical layout information
 * within the station structure.
 *
 * Note:
 * - Station x,y,z coorinates used by this function are assumed to be in
 * wavenumber units.
 * - The \p work_weights buffer must be allocated on the GPU of complex type
 * matching the same floating point precision as the rest of the memory
 * passed to the function.
 *
 *
 * @param[out] beam         Array of station complex beam amplitudes returned.
 * @param[in]  station      Station model structure.
 * @param[in]  hor_l_beam   Beam phase centre horizontal l, in radians.
 * @param[in]  hor_m_beam   Beam phase centre horizontal m, in radians.
 * @param[in]  hor_l        Array of horizontal l directions for which the beam
 *                          should be evaluated, in radians.
 * @param[in]  hor_m        Array of horizontal m directions for which the beam
 *                          should be evaluated, in radians.
 * @param[in]  work_weights Work buffer used to hold DFT weights.
 *
 * @return An error code.
 */
OSKAR_EXPORT
int oskar_evaluate_station_beam(oskar_Mem* beam,
        const oskar_StationModel* station, const double hor_l_beam,
        const double hor_m_beam, const oskar_Mem* hor_l,
        const oskar_Mem* hor_m, oskar_Mem* weights_work);














/**
 * DEPRECATED
 * @brief
 * Function to evaluate the scalar E-Jones (beam pattern) for each source
 * in a local sky model for a specified station array geometry.
 *
 * @details
 * - The arrays d_weights_work and d_e_jones must be pre-allocated on the
 *   device to the following sizes:
 *      - d_weights_work: hd_station.num_antennas * sizeof(double2)
 *      - d_e_jones:      hd_sky.num_sources * sizeof(double2)
 *
 * - The beam phase centre coordinates are specified in horizontal lm
 *   and these can be converted from ra, dec using the function:
 *      - evaluate_beam_horizontal_lm() (FIXME: this currently only exists as a private function in oskar_interferometer1_scalar)
 *
 *
 * @param[in]  hd_station       Pointer to host structure containing device
 *                              pointers to the station antenna coordinates.
 * @param[in]  h_beam_l         Beam phase centre coordinate in horizontal lm
 *                              units.
 * @param[in]  h_beam_m         Beam phase centre coordinate in horizontal lm
 *                              units.
 * @param[in]  hd_sky           Pointer to host structure holding device pointers
 *                              to the local sky model.
 * @param[in]  d_weights_work   Work array for beamforming phase weights.
 * @param[out] d_e_jones        Array containing the E-Jones evaluated for each
 *                              source position.
 *
 * @return CUDA error code.
 */
OSKAR_EXPORT
int oskar_evaluate_station_beam_d(const oskar_StationModel_d* hd_station,
        const double h_beam_l, const double h_beam_m,
        const oskar_SkyModelLocal_d* hd_sky, double2* d_weights_work,
        double2* d_e_jones);

/**
 * DEPRECATED
 * @brief
 * Single precision version of function above.
 */
OSKAR_EXPORT
int oskar_evaluate_station_beam_f(const oskar_StationModel_f* hd_station,
        const float h_beam_l, const float h_beam_m,
        const oskar_SkyModelLocal_f* hd_sky, float2* d_weights_work,
        float2* d_e_jones);

/**
 * DEPRECATED
 * @brief
 * Evaluates the station beam response for a local sky model for a number of
 * stations.
 *
 * @param[in]  num_stations          The number of stations evaluated.
 * @param[in]  hd_stations           Array of structures holding station coordinates.
 * @param[in]  hd_sky                Structure holding the local sky model.
 * @param[in]  h_beam_l              Beam phase centre coordinate in horizontal
 *                                   lm units.
 * @param[in]  h_beam_m              Beam phase centre coordinate in horizontal
 *                                   lm units.
 * @param[in]  d_weights_work        Work buffer for beamforming weights.
 * @param[in]  disable               Flag to disable evaulation of the station
 *                                   beams.
 * @param[in]  identical_stations    Flag to indicate that all stations share
 *                                   identical layouts.
 * @param[out] d_e_jones             Array of beam pattern values.
 */
OSKAR_EXPORT
void oskar_evaluate_station_beams_d(const unsigned num_stations,
        const oskar_StationModel_d* hd_stations,
        const oskar_SkyModelLocal_d* hd_sky, const double h_beam_l,
        const double h_beam_m, double2* d_weights_work,
        bool disable, bool identical_stations, double2* d_e_jones);

/**
 * DEPRECATED
 * @brief
 * Single precision version of function above.
 */
OSKAR_EXPORT
void oskar_evaluate_station_beams_f(const unsigned num_stations,
        const oskar_StationModel_f* hd_stations,
        const oskar_SkyModelLocal_f* hd_sky, const float h_beam_l,
        const float h_beam_m, float2* d_weights_work,
        bool disable, bool identical_stations, float2* d_e_jones);


#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_STATION_BEAM_H_ */
