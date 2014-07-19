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

#include <oskar_evaluate_pierce_points.h>
#include <oskar_convert_ecef_to_geodetic_spherical.h>
#include <math.h>
#include <stdio.h>

static void create_rot_matrix(double* R, double lon, double lat);
static void matrix_multiply(double* v_out, double* M, double* v_in);

#ifdef __cplusplus
extern "C" {
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

void oskar_evaluate_pierce_points(
        oskar_Mem* pierce_point_lon,
        oskar_Mem* pierce_point_lat,
        oskar_Mem* relative_path_length,
        double station_lon,
        double station_lat,
        double station_alt,
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
    double norm_xyz, earth_radius_m;
    double x, y, z;
    int i, type;
    double rotM[9];
    double diff_vector_ENU[3];
    double diff_vector_ECEF[3];
    double scale, arg, alpha_prime, sin_beta;
    double pp_x, pp_y, pp_z;
    double pp_lon, pp_lat, pp_sec, pp_alt;

    /* Check all inputs. */
    if (!pierce_point_lon || !pierce_point_lat || !hor_x || !hor_y || !hor_z)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check memory location (CPU (host) memory current required) */
    if (oskar_mem_location(pierce_point_lat) != OSKAR_CPU ||
            oskar_mem_location(pierce_point_lon) != OSKAR_CPU ||
            oskar_mem_location(relative_path_length) != OSKAR_CPU ||
            oskar_mem_location(hor_x) != OSKAR_CPU ||
            oskar_mem_location(hor_y) != OSKAR_CPU ||
            oskar_mem_location(hor_z) != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    /* Check type consistency */
    type = oskar_mem_type(hor_x);
    if (oskar_mem_type(pierce_point_lat) != type ||
            oskar_mem_type(pierce_point_lon) != type ||
            oskar_mem_type(relative_path_length) != type ||
            oskar_mem_type(hor_y) != type ||
            oskar_mem_type(hor_z) != type)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* Check array size consistency */
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


    /* Pierce point evaluation */
    /* -----------------------------------------------------------------------*/

    /* Length of the vector from the centre of the earth to the station. */
    norm_xyz = sqrt(station_x_ecef*station_x_ecef +
            station_y_ecef*station_y_ecef +
            station_z_ecef*station_z_ecef);

    /* Evaluate the earth radius at the station position. */
    earth_radius_m = norm_xyz - station_alt;

    /* Evaluate a rotation matrix used for ENU to ECEF coordinate conversion. */
    create_rot_matrix(rotM, station_lon, station_lat);

    /* Loop over directions to evaluate pierce points. */
    for (i = 0; i < num_directions; ++i)
    {
        /* Unit vector describing the direction of the pierce point in the
           ENU frame. Vector from station to pierce point. */
        if (type == OSKAR_DOUBLE)
        {
            x = oskar_mem_double_const(hor_x, status)[i];
            y = oskar_mem_double_const(hor_y, status)[i];
            z = oskar_mem_double_const(hor_z, status)[i];
        }
        else
        {
            x = (double)oskar_mem_float_const(hor_x, status)[i];
            y = (double)oskar_mem_float_const(hor_y, status)[i];
            z = (double)oskar_mem_float_const(hor_z, status)[i];
        }
        diff_vector_ENU[0] = x;
        diff_vector_ENU[1] = y;
        diff_vector_ENU[2] = z;

        /* Convert unit vector to ECEF frame. */
        matrix_multiply(diff_vector_ECEF, rotM, diff_vector_ENU);

        /* Evaluate the length of the vector between the station and
           the pierce point. */
        scale = screen_height_m;
        pp_sec = 1.0;

        /* If the direction is directly towards the zenith we don't have to
           calculate anything as the length is simply the screen height! */
        if (fabs(diff_vector_ENU[2] - 1.0) > 1.0e-10)
        {
            double el, cos_el;
            el = asin(z); /* FIXME check this way of evaluating elevation is stable */
            cos_el = cos(el);
            arg = (cos_el * norm_xyz) / (earth_radius_m + screen_height_m);
            alpha_prime = asin(arg);
            pp_sec = 1.0/cos(alpha_prime);
            sin_beta = sin((0.5*M_PI-el)-alpha_prime);
            scale = (earth_radius_m + screen_height_m) * sin_beta / cos_el;
        }

        /* Evaluate the pierce point in ECEF x,y,z coordinates. */
        pp_x = station_x_ecef + (diff_vector_ECEF[0] * scale);
        pp_y = station_y_ecef + (diff_vector_ECEF[1] * scale);
        pp_z = station_z_ecef + (diff_vector_ECEF[2] * scale);

        /* Convert ECEF x,y,z coordinates to long., lat. */
        oskar_convert_ecef_to_geodetic_spherical(1, &pp_x, &pp_y, &pp_z,
                &pp_lon, &pp_lat, &pp_alt);

        if (type == OSKAR_DOUBLE)
        {
            oskar_mem_double(pierce_point_lon, status)[i] = pp_lon;
            oskar_mem_double(pierce_point_lat, status)[i] = pp_lat;
            oskar_mem_double(relative_path_length, status)[i] = pp_sec;
        }
        else
        {
            oskar_mem_float(pierce_point_lon, status)[i] = (float)pp_lon;
            oskar_mem_float(pierce_point_lat, status)[i] = (float)pp_lat;
            oskar_mem_float(relative_path_length, status)[i] = (float)pp_sec;
        }
    } /* loop over p.p. directions. */
}



/* Populate a rotation matrix for ENU to ITRF conversion.
   3x3 matrix, row-major order. */
void create_rot_matrix(double* R, double lon, double lat)
{
    double cosl   = cos(lon);
    double sinl   = sin(lon);
    double cosphi = cos(lat);
    double sinphi = sin(lat);
    R[0] = -sinl;
    R[1] = -sinphi * cosl;
    R[2] =  cosphi * cosl;
    R[3] =  cosl;
    R[4] = -sinphi * sinl;
    R[5] =  cosphi * sinl;
    R[6] =  0.0;
    R[7] =  cosphi;
    R[8] =  sinphi;
}

/* row-major matrix vector multiply. */
void matrix_multiply(double* v_out, double* M, double* v_in)
{
    v_out[0] = (M[0] * v_in[0]) + (M[1] * v_in[1]) + (M[2] * v_in[2]);
    v_out[1] = (M[3] * v_in[0]) + (M[4] * v_in[1]) + (M[5] * v_in[2]);
    v_out[2] = (M[6] * v_in[0]) + (M[7] * v_in[1]) + (M[8] * v_in[2]);
}

#ifdef __cplusplus
}
#endif
