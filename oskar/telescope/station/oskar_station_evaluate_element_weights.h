/*
 * Copyright (c) 2012-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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
 * @param[in] frequency_hz        Observing frequency, in Hz.
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
        int feed, double frequency_hz, double x_beam, double y_beam,
        double z_beam, int time_index, oskar_Mem* weights,
        oskar_Mem* weights_scratch, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
