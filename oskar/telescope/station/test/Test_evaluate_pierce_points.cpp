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

#include <gtest/gtest.h>

#include "convert/oskar_convert_enu_to_ecef.h"
#include "telescope/station/oskar_evaluate_pierce_points.h"
#include "convert/oskar_convert_ecef_to_geodetic_spherical.h"

#include "mem/oskar_mem.h"

#include "math/oskar_cmath.h"
#include <cstdio>

static void create_rot_matrix(double* M, double lon_rad, double lat_rad);
static void matrix_multiply(double* v_out, double* M, double *v_in);

TEST(evaluate_pierce_points, test1)
{
    // ====== INPUTS =========================================================

    // Station lon/lat coordinates (for which pierce points are evaluated)
    double st_lon_deg = 0.0;
    double st_lat_deg = 45.0;
    double st_alt_m   = 0.0;

    // Source position.
    double az_deg = 0.0;
    double el_deg = 80.0;
    // Ionosphere.
    double height_km = 300;

    // Station co-ordinates (horizontal x,y,z - as in 'layout' file).
    double st_hor_x = 0.0;
    double st_hor_y = 0.0;
    double st_hor_z = 0.0;

    // Notes:
    // - based on make_pp from PiercePoints.py (meqtrees-cattery/Lions)

    // ========================================================================

    double deg2rad = M_PI/180.0;
    double st_lon_rad = st_lon_deg * deg2rad;
    double st_lat_rad = st_lat_deg * deg2rad;
    double el_rad = el_deg * deg2rad;
    double az_rad = az_deg * deg2rad;
    double height_m = height_km * 1000.0;

    // == Evaluate geo-centric station x,y,z coordinates.
    double st_x, st_y, st_z;
    oskar_convert_enu_to_ecef(1, &st_hor_x, &st_hor_y, &st_hor_z,
            st_lon_rad, st_lat_rad, st_alt_m, &st_x, &st_y, &st_z);
    printf("  lon = %f, lat = %f [station]\n", st_lon_deg, st_lat_deg);
    printf("  az = %f, el = %f [station]\n", az_deg, el_deg);
    printf("  geocentric cartesian: x=%f, y=%f, z=%f\n", st_x, st_y, st_z);

    // == Create rot_matrix.
    double rot_matrix[9];
    create_rot_matrix(rot_matrix, st_lon_rad, st_lat_rad);

    // == station, source loop (ref: line 58)
    double norm_xyz = sqrt(st_x*st_x + st_y*st_y + st_z*st_z);
    double earth_radius_m = norm_xyz - st_alt_m;
    printf("  norm_xzy = %f\n", norm_xyz);
    printf("  earth_radius_m = %f\n", earth_radius_m);
    double cos_az = cos(az_rad);
    double sin_az = sin(az_rad);
    double cos_el = cos(el_rad);
    double sin_el = sin(el_rad);

    // Evaluate unit diff vector in ENU frame and convert
    // to geocentric frame.
    // this is hor{_x,_y,_z}
    double diff_vector_ENU[3] =
    { cos_el * sin_az, cos_el * cos_az, sin_el };

    printf("  diff vector ENU = %f, %f, %f\n",
            diff_vector_ENU[0],
            diff_vector_ENU[1],
            diff_vector_ENU[2]);

    double diff_vector_ITRF[3];
    matrix_multiply(diff_vector_ITRF, rot_matrix, diff_vector_ENU);

    printf("  diff vector ITRF = %f, %f, %f\n",
            diff_vector_ITRF[0],
            diff_vector_ITRF[1],
            diff_vector_ITRF[2]);

    double scale = height_m;

    printf("  cos_el = %e, %e\n", cos_el, fabs(cos_el));
    if (fabs(cos_el) > 1.0e-10)
    {
        double arg = (cos_el * norm_xyz) / (earth_radius_m + height_m);
        double alpha_prime = asin(arg);
        printf("  arg = %f\n", arg);
        printf("  alpha_prime = %f\n", alpha_prime/deg2rad);

        double sin_beta = sin((0.5*M_PI - el_rad) - alpha_prime);
        printf("  sin_beta = %f\n", sin_beta/deg2rad);

        scale = (earth_radius_m + height_m) * sin_beta / cos_el;
        printf("  scale = %f km\n", scale/1000.);
    }

    double pp_x = st_x + (diff_vector_ITRF[0] * scale);
    double pp_y = st_y + (diff_vector_ITRF[1] * scale);
    double pp_z = st_z + (diff_vector_ITRF[2] * scale);

    double pp_lon, pp_lat, pp_alt;
    oskar_convert_ecef_to_geodetic_spherical(1, &pp_x, &pp_y, &pp_z,
            &pp_lon, &pp_lat, &pp_alt);

    printf("  pierce point: x=%f, y=%f, z=%f\n", pp_x, pp_y, pp_z);
//    double norm_pp = sqrt(pp_x*pp_x+pp_y*pp_y+pp_z*pp_z);
    printf("  pierce point: lon=%f, lat=%f (alt=%f km)\n",
            pp_lon*180.0/M_PI,
            pp_lat*180./M_PI,
            pp_alt/1000.);
}


TEST(evaluate_pierce_points, test2)
{
    // >>>>>> Inputs. <<<<<<<<<

    // Pierce point settings (screen height, az, el of source)
    double az     = 0.0 * (M_PI/180.);
    double el     = 80.0 * (M_PI/180.);
    double height = 300. * 1000.;

    // Station position (longitude, latitude, altitude)
    double lon = 0.0 * (M_PI/180.);
    double lat = 45.0 * (M_PI/180.);
    double alt = 0.0;

    // Station horizontal x,y,z as in the layout file.
    double st_hor_x = 0.0;
    double st_hor_y = 0.0;
    double st_hor_z = 0.0;

    // Obtain station ECEF coordinates (geocentric x,y,z coordinates of station)
    double x, y, z;
    oskar_convert_enu_to_ecef(1, &st_hor_x, &st_hor_y, &st_hor_z, lon, lat,
            alt, &x, &y, &z);

    // Evaluate horizontal x,y,z of the pierce point.
    int n = 1;
    oskar_Mem *hor_x, *hor_y, *hor_z;
    int type = OSKAR_DOUBLE;
    int location = OSKAR_CPU;
    int status = 0;
    hor_x = oskar_mem_create(type, location, n, &status);
    hor_y = oskar_mem_create(type, location, n, &status);
    hor_z = oskar_mem_create(type, location, n, &status);
    double x_ = cos(el) * sin(az);
    double y_ = cos(el) * cos(az);
    double z_ = sin(el);
    oskar_mem_double(hor_x, &status)[0] = x_;
    oskar_mem_double(hor_y, &status)[0] = y_;
    oskar_mem_double(hor_z, &status)[0] = z_;

    // Evaluate the pierce points.
    oskar_Mem *pp_lon, *pp_lat, *pp_path;
    pp_lon = oskar_mem_create(type, location, n, &status);
    pp_lat = oskar_mem_create(type, location, n, &status);
    pp_path = oskar_mem_create(type, location, n, &status);
    oskar_evaluate_pierce_points(pp_lon, pp_lat, pp_path, x, y, z,
            height, n, hor_x, hor_y, hor_z, &status);

    printf("pierce point [%i]:\n", 0);
    printf("  lon = %f, lat = %f [station]\n", lon*(180./M_PI), lat*(180./M_PI));
    printf("  x = %f, y = %f, z = %f [station]\n", x, y, z);
    printf("  hor_x = %f, hor_y = %f, hor_z = %f\n", x_, y_, z_);
    printf("  az = %f, el = %f\n", az*(180./M_PI), el*(180./M_PI));
    printf("  lon=%f, lat=%f, path=%f\n",
            oskar_mem_double(pp_lon, &status)[0]*(180./M_PI),
            oskar_mem_double(pp_lat, &status)[0]*(180./M_PI),
            oskar_mem_double(pp_path, &status)[0]);

    // Free memory.
    oskar_mem_free(hor_x, &status);
    oskar_mem_free(hor_y, &status);
    oskar_mem_free(hor_z, &status);
    oskar_mem_free(pp_lon, &status);
    oskar_mem_free(pp_lat, &status);
    oskar_mem_free(pp_path, &status);
}


static void create_rot_matrix(double* M, double lon_rad, double lat_rad)
{
    double cosl   = cos(lon_rad);
    double sinl   = sin(lon_rad);
    double cosphi = cos(lat_rad);
    double sinphi = sin(lat_rad);

// Meqtrees is column major! keep this commented out if matrix mult. is C order!
//    M[0] = -sinl;
//    M[1] = cosl;
//    M[2] = 0;
//    M[3] = -sinphi*cosl;
//    M[4] = -sinphi*sinl;
//    M[5] = cosphi;
//    M[6] = cosphi * cosl;
//    M[7] = cosphi * sinl;
//    M[8] = sinphi;

    M[0] = -sinl;
    M[1] = -sinphi*cosl;
    M[2] = cosphi * cosl;
    M[3] = cosl;
    M[4] = -sinphi*sinl;
    M[5] = cosphi * sinl;
    M[6] = 0;
    M[7] = cosphi;
    M[8] = sinphi;
}

static void matrix_multiply(double* v_out, double* M, double *v_in)
{
    v_out[0] = (M[0] * v_in[0]) + (M[1] * v_in[1]) + (M[2] * v_in[2]);
    v_out[1] = (M[3] * v_in[0]) + (M[4] * v_in[1]) + (M[5] * v_in[2]);
    v_out[2] = (M[6] * v_in[0]) + (M[7] * v_in[1]) + (M[8] * v_in[2]);
}

