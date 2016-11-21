/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#ifndef OSKAR_EVALUATE_STATION_BEAM_APERTURE_ARRAY_H_
#define OSKAR_EVALUATE_STATION_BEAM_APERTURE_ARRAY_H_

/**
 * @file oskar_evaluate_station_beam_aperture_array.h
 */

#include <oskar_global.h>
#include <oskar_mem.h>
#include <oskar_station.h>
#include <oskar_station_work.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates the station beam for an aperture array station.
 *
 * @details
 * This function evaluates the beam for an aperture array station,
 * including any possible child stations and/or element models within the
 * station.
 *
 * The supplied station model must be fully set-up, and must include
 * the element positions, the required beam coordinates, and all associated
 * options and meta-data necessary for beamforming.
 *
 * The beam is evaluated at points defined by the horizontal Cartesian
 * direction cosines given in the vectors x,y,z. The output \p beam array
 * must be of the correct data type for the station element:
 * for example, it is an error to use polarised elements and supply a scalar
 * type for the output data. If in doubt, an output type of complex matrix will
 * always be acceptable, although it may be significantly slower for isotropic
 * elements.
 *
 * The work structure holds pointers to memory blocks used by routines which
 * are called by this wrapper. The structure must be initialised, but may be
 * empty. In this case, the internal buffers will be resized to the correct
 * dimensions on first use. Subsequent calls to this function should ideally
 * re-use the same work structure to ensure optimum performance and no needless
 * memory reallocation.
 *
 * @param[out]    beam          Station beam evaluated at x,y,z positions.
 * @param[in]     station       Fully populated station model structure.
 * @param[in]     num_points    Number of coordinates at which to evaluate
 *                              the beam.
 * @param[in]     x             Array of horizontal x coordinates at which to
 *                              evaluate the beam.
 * @param[in]     y             Array of horizontal y coordinates at which to
 *                              evaluate the beam.
 * @param[in]     z             Array of horizontal z coordinates at which to
 *                              evaluate the beam.
 * @param[in]     gast          The Greenwich Apparent Sidereal Time in radians.
 * @param[in]     frequency_hz  The observing frequency, in Hz.
 * @param[in]     work          Initialised structure containing temporary work
 *                              buffers.
 * @param[in]     time_index    Simulation time index.
 * @param[in,out] status        Status return code.
 */
OSKAR_EXPORT
void oskar_evaluate_station_beam_aperture_array(oskar_Mem* beam,
        const oskar_Station* station, int num_points, const oskar_Mem* x,
        const oskar_Mem* y, const oskar_Mem* z, double gast,
        double frequency_hz, oskar_StationWork* work, int time_index,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_STATION_BEAM_APERTURE_ARRAY_H_ */
