/*
 * Copyright (c) 2011-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_EVALUATE_JONES_E_H_
#define OSKAR_EVALUATE_JONES_E_H_

/**
 * @file oskar_evaluate_jones_E.h
 */

#include <oskar_global.h>
#include <telescope/oskar_telescope.h>
#include <interferometer/oskar_jones.h>

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
 * @param[out] E             Output set of Jones matrices.
 * @param[in]  coord_type    Type of coordinates.
 * @param[in]  num_points    Number of coordinates given.
 * @param[in]  source_coords Source coordinate values.
 * @param[in]  ref_lon_rad   Reference longitude in radians, if inputs are direction cosines.
 * @param[in]  ref_lat_rad   Reference latitude in radians, if inputs are direction cosines.
 * @param[in]  tel           Input telescope model.
 * @param[in]  time_index    Simulation time index.
 * @param[in]  gast_rad      The Greenwich Apparent Sidereal Time, in radians.
 * @param[in]  frequency_hz  The observing frequency, in Hz.
 * @param[in]  work          Pointer to structure holding work arrays.
 * @param[in,out] status     Status return code.
 */
OSKAR_EXPORT
void oskar_evaluate_jones_E(
        oskar_Jones* E,
        int coord_type,
        int num_points,
        const oskar_Mem* const source_coords[3],
        double ref_lon_rad,
        double ref_lat_rad,
        const oskar_Telescope* tel,
        int time_index,
        double gast_rad,
        double frequency_hz,
        oskar_StationWork* work,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
