/*
 * Copyright (c) 2012-2020, The University of Oxford
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

#ifndef OSKAR_STATION_EVALUATE_ELEMENT_WEIGHTS_H_
#define OSKAR_STATION_EVALUATE_ELEMENT_WEIGHTS_H_

/**
 * @file oskar_station_evaluate_element_weights.h
 */

#include <oskar_global.h>
#include <telescope/station/oskar_station.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Top-level function to evaluate element beamforming weights for the station.
 *
 * @details
 * This function evaluates the required element beamforming weights.
 * These weights are a combination of:
 *
 * - Geometric DFT phases.
 * - Systematic and random gain and phase variations.
 * - User-supplied apodisation weights.
 *
 * The \p weights and \p weights_scratch arrays are resized to hold the weights
 * if necessary.
 *
 * @param[in] station             Station model.
 * @param[in] feed                Feed index (0 = X, 1 = Y).
 * @param[in] wavenumber          Wavenumber (2 pi / wavelength).
 * @param[in] x_beam              Beam direction cosine, horizontal x-component.
 * @param[in] y_beam              Beam direction cosine, horizontal y-component.
 * @param[in] z_beam              Beam direction cosine, horizontal z-component.
 * @param[in] time_index          Time index of simulation.
 * @param[in,out] weights         Output array of beamforming weights.
 * @param[in,out] weights_scratch Work array, for calculating the weights error.
 * @param[in,out] status          Status return code.
 */
OSKAR_EXPORT
void oskar_station_evaluate_element_weights(const oskar_Station* station,
        int feed, double wavenumber, double x_beam, double y_beam,
        double z_beam, int time_index, oskar_Mem* weights,
        oskar_Mem* weights_scratch, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
