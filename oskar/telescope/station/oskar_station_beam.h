/*
 * Copyright (c) 2013-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_STATION_BEAM_H_
#define OSKAR_STATION_BEAM_H_

/**
 * @file oskar_station_beam.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluate the beam for a station.
 *
 * @details
 * Evaluates the beam of a station at the specified positions.
 *
 * @param[in] station           Station model.
 * @param[in] work              Station beam workspace.
 * @param[in] source_coord_type Type of input/source coordinates
 *                              (OSKAR_COORD_TYPE enumerator).
 * @param[in] num_points        Number of points at which to evaluate beam.
 * @param[in] source_coords     Source coordinate values.
 * @param[in] ref_lon_rad       Reference longitude in radians,
 *                              if inputs are direction cosines.
 * @param[in] ref_lat_rad       Reference latitude in radians,
 *                              if inputs are direction cosines.
 * @param[in] norm_coord_type   Type of normalisation coordinates.
 * @param[in] norm_lon_rad      Longitude for beam normalisation, in radians.
 * @param[in] norm_lat_rad      Latitude for beam normalisation, in radians.
 * @param[in] time_index        Simulation time index.
 * @param[in] gast_rad          Greenwich Apparent Sidereal Time, in radians.
 * @param[in] frequency_hz      The observing frequency in Hz.
 * @param[in] offset_out        Output array element offset.
 * @param[out] beam             Output beam data.
 * @param[in,out] status        Status return code.
 */
OSKAR_EXPORT
void oskar_station_beam(
        const oskar_Station* station,
        oskar_StationWork* work,
        int source_coord_type,
        int num_points,
        const oskar_Mem* const source_coords[3],
        double ref_lon_rad,
        double ref_lat_rad,
        int norm_coord_type,
        double norm_lon_rad,
        double norm_lat_rad,
        int time_index,
        double gast_rad,
        double frequency_hz,
        int offset_out,
        oskar_Mem* beam,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
