/*
 * Copyright (c) 2013-2014, The University of Oxford
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

#ifndef OSKAR_EVALUATE_PIERCE_POINTS_H_
#define OSKAR_EVALUATE_PIERCE_POINTS_H_

/**
 * @file oskar_evaluate_pierce_points.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Evaluates pierce points for a station for a number of directions.
 *
 * @details
 * This function is based on the MeqTrees script Lions/PiercePoints.py
 * which was developed by Maaijke Mevius and Ilse van Bemmel, and
 * can be found in the MeqTrees cattery repository.
 *
 * Possible problems:
 * - Pierce points below the horizon are still evaluated.
 *
 * @param[out] pierce_point_lon      Array of pierce point longitudes, in radians.
 * @param[out] pierce_point_lat      Array of pierce point latitudes, in radians.
 * @param[out] relative_path_length  Array of relative path lengths in the direction
 *                                   of the pierce point from the station
 *                                   [sec(alpha_prime)].
 * @param[in] station_x_ecef    Station x (earth centred, earth fixed
 *                              /ITRF/geocentric) coordinate.
 * @param[in] station_x_ecef    Station y (earth centred, earth fixed
 *                              /ITRF/geocentric) coordinate.
 * @param[in] station_z_ecef    Station z (earth centred, earth fixed
 *                              /ITRF/geocentric) coordinate.
 * @param[in] screen_height_m   Height of the screen though which to calculate
 *                              pierce points, in metres.
 * @param[in] num_directions    Number of directions for which to calculate
 *                              pierce points.
 * @param[in] hor_x             Array of horizontal x direction cosines for
 *                              which to evaluate pierce points.
 * @param[in] hor_y             Array of horizontal y direction cosines for
 *                              which to evaluate pierce points.
 * @param[in] hor_z             Array of horizontal z direction cosines for
 *                              which to evaluate pierce points.
 * @param[in,out] status        Error status code.
 */
OSKAR_EXPORT
void oskar_evaluate_pierce_points(
        oskar_Mem* pierce_point_lon,
        oskar_Mem* pierce_point_lat,
        oskar_Mem* relative_path_length,
        double station_x_ecef,
        double station_y_ecef,
        double station_z_ecef,
        double screen_height_m,
        int num_directions,
        const oskar_Mem* hor_x,
        const oskar_Mem* hor_y,
        const oskar_Mem* hor_z,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_PIERCE_POINTS_H_ */
