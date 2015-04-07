/*
 * Copyright (c) 2011-2015, The University of Oxford
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

#ifndef OSKAR_EVALUATE_JONES_E_H_
#define OSKAR_EVALUATE_JONES_E_H_

/**
 * @file oskar_evaluate_jones_E.h
 */

#include <oskar_global.h>
#include <oskar_telescope.h>
#include <oskar_jones.h>
#include <oskar_station_work.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates a set of E-Jones matrices for a number of stations and
 * source positions.
 *
 * @details
 * Evaluates station beams for a telescope model at the specified source
 * positions, storing the results in the Jones matrix data structure.
 *
 * If all stations are marked as identical, the results for the first station
 * are copied into the results for the others.
 *
 * @param[out] E            Output set of Jones matrices.
 * @param[in]  num_points   Number of direction cosines given.
 * @param[in]  coord_type   Type of direction cosines
 *                          (OSKAR_RELATIVE_DIRECTIONS or
 *                          OSKAR_ENU_DIRECTIONS).
 * @param[in]  x            Direction cosines (x direction).
 * @param[in]  y            Direction cosines (y direction).
 * @param[in]  z            Direction cosines (z direction).
 * @param[in]  tel          Input telescope model.
 * @param[in]  gast         The Greenwich Apparent Sidereal Time, in radians.
 * @param[in]  frequency_hz The observing frequency, in Hz.
 * @param[in]  work         Pointer to structure holding work arrays.
 * @param[in]  time_index   Simulation time index.
 * @param[in,out] status    Status return code.
 */
OSKAR_EXPORT
void oskar_evaluate_jones_E(oskar_Jones* E, int num_points, int coord_type,
        oskar_Mem* x, oskar_Mem* y, oskar_Mem* z, const oskar_Telescope* tel,
        double gast, double frequency_hz, oskar_StationWork* work,
        int time_index, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_JONES_E_H_ */
