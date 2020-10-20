/*
 * Copyright (c) 2012-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_EVALUATE_STATION_BEAM_APERTURE_ARRAY_H_
#define OSKAR_EVALUATE_STATION_BEAM_APERTURE_ARRAY_H_

/**
 * @file oskar_evaluate_station_beam_aperture_array.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>
#include <telescope/station/oskar_station.h>

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
 * direction cosines given in the vectors x,y,z.
 *
 * @param[in]     station       Fully populated station model structure.
 * @param[in]     work          Station beam workspace.
 * @param[in]     num_points    Number of coordinates at which to evaluate
 *                              the beam.
 * @param[in]     x             Array of horizontal x coordinates at which to
 *                              evaluate the beam.
 * @param[in]     y             Array of horizontal y coordinates at which to
 *                              evaluate the beam.
 * @param[in]     z             Array of horizontal z coordinates at which to
 *                              evaluate the beam.
 * @param[in]     time_index    Simulation time index.
 * @param[in]     gast_rad      Greenwich Apparent Sidereal Time in radians.
 * @param[in]     frequency_hz  The observing frequency, in Hz.
 * @param[out]    beam          Station beam evaluated at x,y,z positions.
 * @param[in,out] status        Status return code.
 */
OSKAR_EXPORT
void oskar_evaluate_station_beam_aperture_array(
        const oskar_Station* station,
        oskar_StationWork* work,
        int num_points,
        const oskar_Mem* x,
        const oskar_Mem* y,
        const oskar_Mem* z,
        int time_index,
        double gast_rad,
        double frequency_hz,
        oskar_Mem* beam,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
