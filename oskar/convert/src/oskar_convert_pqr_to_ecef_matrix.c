/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <math.h>
#include "convert/oskar_convert_pqr_to_ecef_matrix.h"
#include "convert/oskar_convert_ecef_to_geodetic_spherical.h"

#ifdef __cplusplus
extern "C" {
#endif


/* 3D vector cross-product: out = a x b. */
static void cross(const double a[3], const double b[3], double out[3])
{
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}


/*
 * 3D matrix-matrix multiply: out = a * b.
 * Matrices are in row-major ("C") order.
 */
static void matrix_mul(const double a[9], const double b[9], double out[9])
{
    for (int row = 0; row < 3; ++row) /* Row for A and output. */
    {
        for (int col = 0; col < 3; ++col) /* Column for B and output. */
        {
            out[row * 3 + col] = (
                    a[row * 3 + 0] * b[0 * 3 + col] +
                    a[row * 3 + 1] * b[1 * 3 + col] +
                    a[row * 3 + 2] * b[2 * 3 + col]
            );
        }
    }
}


/* Normalise length of 3D vector. */
static void norm(double v[3])
{
    const double scale = 1.0 / sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    v[0] *= scale;
    v[1] *= scale;
    v[2] *= scale;
}


static void normal_vector_meridian_plane(const double ecef[3], double out[3])
{
    const double x = ecef[0], y = ecef[1], scale = 1.0 / sqrt(x*x + y*y);
    out[0] = y * scale;
    out[1] = -x * scale;
    out[2] = 0.0;
}


/* Calculate normal vector to Earth ellipsoid at a station position. */
void oskar_convert_ecef_to_ellipsoid_normal(
        const double station_ecef_coords[3],
        double norm_vec_ellipsoid[3]
)
{
    double station_wgs84[3];

    /* Get longitude, latitude, altitude of the station. */
    oskar_convert_ecef_to_geodetic_spherical(
            1,
            &station_ecef_coords[0],
            &station_ecef_coords[1],
            &station_ecef_coords[2],
            &station_wgs84[0],
            &station_wgs84[1],
            &station_wgs84[2]
    );

    /* Get station vector from longitude and latitude. */
    norm_vec_ellipsoid[0] = cos(station_wgs84[1]) * cos(station_wgs84[0]);
    norm_vec_ellipsoid[1] = cos(station_wgs84[1]) * sin(station_wgs84[0]);
    norm_vec_ellipsoid[2] = sin(station_wgs84[1]);
}


/*
 * Following exact method in lofarantpos-0.4.1 for consistency.
 * The equivalent function in lofarantpos is called "projection_matrix".
 */
void oskar_convert_enu_to_ecef_matrix(
        const double station_ecef_coords[3],
        const double norm_vec[3],
        double m[9]
)
{
    double p_unit[3], q_unit[3], meridian_normal[3];
    const double r_unit[3] = {norm_vec[0], norm_vec[1], norm_vec[2]};
    normal_vector_meridian_plane(station_ecef_coords, meridian_normal);
    cross(meridian_normal, r_unit, q_unit);
    norm(q_unit);
    cross(q_unit, r_unit, p_unit);
    norm(p_unit);
    m[0] = p_unit[0]; m[1] = q_unit[0]; m[2] = r_unit[0];
    m[3] = p_unit[1]; m[4] = q_unit[1]; m[5] = r_unit[1];
    m[6] = p_unit[2]; m[7] = q_unit[2]; m[8] = r_unit[2];
}


/*
 * Calculate matrix for conversion of PQR to ENU coordinates.
 * This is a rotation around the zenith by the station angle.
 */
void oskar_convert_pqr_to_enu_matrix(
        const double station_rotation_angle_rad,
        double pqr_to_enu[9]
)
{
    const double sin_angle = sin(station_rotation_angle_rad);
    const double cos_angle = cos(station_rotation_angle_rad);
    pqr_to_enu[0] = cos_angle;  pqr_to_enu[1] = sin_angle; pqr_to_enu[2] = 0.;
    pqr_to_enu[3] = -sin_angle; pqr_to_enu[4] = cos_angle; pqr_to_enu[5] = 0.;
    pqr_to_enu[6] = 0.;         pqr_to_enu[7] = 0.;        pqr_to_enu[8] = 1.;
}


void oskar_convert_pqr_to_ecef_matrix(
        const double station_ecef_coords[3],
        const double station_rotation_angle_rad,
        double pqr_to_ecef[9]
)
{
    double enu_to_ecef[9], pqr_to_enu[9];
    double norm_vec_ellipsoid[3];

    /* Get normal vector to Earth ellipsoid at station. */
    oskar_convert_ecef_to_ellipsoid_normal(
            station_ecef_coords, norm_vec_ellipsoid
    );

    /* Get PQR to ENU matrix (rotation around zenith by station angle). */
    oskar_convert_pqr_to_enu_matrix(station_rotation_angle_rad, pqr_to_enu);

    /* Get ENU to ECEF (ITRF) matrix ("projection_matrix" in lofarantpos). */
    oskar_convert_enu_to_ecef_matrix(
            station_ecef_coords, norm_vec_ellipsoid, enu_to_ecef
    );

    /*
     * Multiply matrices to get PQR to ECEF transformation matrix:
     * pqr_to_ecef = enu_to_ecef * pqr_to_enu
     */
    matrix_mul(enu_to_ecef, pqr_to_enu, pqr_to_ecef);
}


#ifdef __cplusplus
}
#endif
