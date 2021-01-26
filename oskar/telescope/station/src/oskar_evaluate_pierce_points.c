/*
 * Copyright (c) 2013-2015, The University of Oxford
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

#include "convert/oskar_convert_ecef_to_geodetic_spherical.h"
#include "math/oskar_cmath.h"
#include "telescope/station/oskar_evaluate_pierce_points.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_evaluate_pierce_points_d(int num_directions, const double* hor_x,
        const double* hor_y, const double* hor_z, double* pp_lon_,
        double* pp_lat_, double* rel_path_len_, double screen_height_m,
        const double station_ecef_x, const double station_ecef_y,
        const double station_ecef_z);

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
        int* status)
{
    int type, location;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check memory location consistency. */
    location = oskar_mem_location(hor_x);
    if (oskar_mem_location(pierce_point_lat) != location ||
            oskar_mem_location(pierce_point_lon) != location ||
            oskar_mem_location(relative_path_length) != location ||
            oskar_mem_location(hor_y) != location ||
            oskar_mem_location(hor_z) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }

    /* Check type consistency. */
    type = oskar_mem_type(hor_x);
    if (oskar_mem_type(pierce_point_lat) != type ||
            oskar_mem_type(pierce_point_lon) != type ||
            oskar_mem_type(relative_path_length) != type ||
            oskar_mem_type(hor_y) != type ||
            oskar_mem_type(hor_z) != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Check array sizes. */
    if ((int)oskar_mem_length(pierce_point_lat) < num_directions ||
            (int)oskar_mem_length(pierce_point_lon) < num_directions ||
            (int)oskar_mem_length(relative_path_length) < num_directions ||
            (int)oskar_mem_length(hor_x) < num_directions ||
            (int)oskar_mem_length(hor_y) < num_directions ||
            (int)oskar_mem_length(hor_z) < num_directions)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Switch on type. */
    if (type == OSKAR_DOUBLE)
    {
        double *pp_lon_, *pp_lat_, *rel_path_len_;
        const double *x_, *y_, *z_;
        x_ = oskar_mem_double_const(hor_x, status);
        y_ = oskar_mem_double_const(hor_y, status);
        z_ = oskar_mem_double_const(hor_z, status);
        pp_lon_ = oskar_mem_double(pierce_point_lon, status);
        pp_lat_ = oskar_mem_double(pierce_point_lat, status);
        rel_path_len_ = oskar_mem_double(relative_path_length, status);

        if (location == OSKAR_CPU)
        {
            oskar_evaluate_pierce_points_d(num_directions, x_, y_, z_,
                    pp_lon_, pp_lat_, rel_path_len_, screen_height_m,
                    station_x_ecef, station_y_ecef, station_z_ecef);
        }
        else
        {
#ifdef OSKAR_HAVE_CUDA
            *status = OSKAR_ERR_BAD_LOCATION;
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
    }
    else
    {
#if 0
        float *pp_lon_, *pp_lat_, *rel_path_len_;
        const float *x_, *y_, *z_;
        x_ = oskar_mem_float_const(hor_x, status);
        y_ = oskar_mem_float_const(hor_y, status);
        z_ = oskar_mem_float_const(hor_z, status);
        pp_lon_ = oskar_mem_float(pierce_point_lon, status);
        pp_lat_ = oskar_mem_float(pierce_point_lat, status);
        rel_path_len_ = oskar_mem_float(relative_path_length, status);
#endif

        if (location == OSKAR_CPU)
        {
            /* oskar_evaluate_pierce_points_f(num_directions, x_, y_, z_,
                    pp_lon_, pp_lat_, rel_path_len_, screen_height_m,
                    station_x_ecef, station_y_ecef, station_z_ecef); */
            *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE; /* FIXME */
        }
        else
        {
#ifdef OSKAR_HAVE_CUDA
            *status = OSKAR_ERR_BAD_LOCATION;
#else
            *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        }
    }
}

void oskar_evaluate_pierce_points_d(int num_directions, const double* hor_x,
        const double* hor_y, const double* hor_z, double* pp_lon_,
        double* pp_lat_, double* rel_path_len_, double screen_height_m,
        const double station_ecef_x, const double station_ecef_y,
        const double station_ecef_z)
{
    int i;
    double norm_xyz, earth_radius_plus_screen_height_m;
    double cos_l, sin_l, cos_b, sin_b, station_lon, station_lat, station_alt;

    /* Get the station longitude and latitude from ECEF coordinates. */
    oskar_convert_ecef_to_geodetic_spherical(1,
            &station_ecef_x, &station_ecef_y, &station_ecef_z,
            &station_lon, &station_lat, &station_alt);

    /* Calculate sine and cosine of station longitude and latitude. */
    sin_l = sin(station_lon);
    cos_l = cos(station_lon);
    sin_b = sin(station_lat);
    cos_b = cos(station_lat);

    /* Length of the vector from the centre of the Earth to the station. */
    norm_xyz = sqrt(station_ecef_x * station_ecef_x +
            station_ecef_y * station_ecef_y +
            station_ecef_z * station_ecef_z);

    /* Evaluate the Earth radius plus screen height at the station position. */
    earth_radius_plus_screen_height_m =
            screen_height_m + norm_xyz - station_alt;

    /* Loop over directions to evaluate pierce points. */
    for (i = 0; i < num_directions; ++i)
    {
        double pp_lon, pp_lat, pp_sec, scale, x, y, z;
        double diff_ecef_x, diff_ecef_y, diff_ecef_z;

        /* Unit vector describing the direction of the pierce point in the
         * ENU frame. Vector from station to pierce point. */
        x = hor_x[i];
        y = hor_y[i];
        z = hor_z[i];

        /* Evaluate length of vector between station and pierce point. */
        /* If the direction is directly towards the zenith we don't have to
         * calculate anything, as the length is simply the screen height! */
        if (fabs(z - 1.0) > 1.0e-10)
        {
            double el, cos_el, arg, alpha_prime, sin_beta;
            el = asin(z);
            cos_el = cos(el);
            arg = (cos_el * norm_xyz) / earth_radius_plus_screen_height_m;
            alpha_prime = asin(arg);
            sin_beta = sin(((M_PI/2) - el) - alpha_prime);
            pp_sec = 1.0 / cos(alpha_prime);
            scale = earth_radius_plus_screen_height_m * sin_beta / cos_el;
        }
        else
        {
            pp_sec = 1.0;
            scale = screen_height_m;
        }

        /* Convert ENU unit vector to ECEF frame using rotation matrix. */
        diff_ecef_x = -x * sin_l - y * sin_b * cos_l + z * cos_b * cos_l;
        diff_ecef_y =  x * cos_l - y * sin_b * sin_l + z * cos_b * sin_l;
        diff_ecef_z =  y * cos_b + z * sin_b;

        /* Evaluate the pierce point in ECEF coordinates. */
        x = station_ecef_x + diff_ecef_x * scale;
        y = station_ecef_y + diff_ecef_y * scale;
        z = station_ecef_z + diff_ecef_z * scale;

        /* Convert ECEF coordinates to geocentric longitude and latitude. */
        pp_lon = atan2(y, x);
        pp_lat = atan2(z, sqrt(x*x + y*y));

        /* Store data. */
        pp_lon_[i] = pp_lon;
        pp_lat_[i] = pp_lat;
        rel_path_len_[i] = pp_sec;
    }
}

#ifdef __cplusplus
}
#endif
