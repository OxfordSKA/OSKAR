/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_CONVERT_PQR_TO_ECEF_MATRIX_H_
#define OSKAR_CONVERT_PQR_TO_ECEF_MATRIX_H_

/**
 * @file oskar_convert_pqr_to_ecef_matrix.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Get conversion matrix from PQR to ECEF frame for PHASED_ARRAY table.
 *
 * @details
 * Calculate the conversion matrix from PQR frame in a rotated station
 * to ECEF frame for PHASED_ARRAY table.
 *
 * The output matrix is in row-major ("C") order.
 *
 * @param[in] station_ecef_coords  ECEF coordinates of station centre.
 * @param[in] station_rotation_angle_rad  Station rotation angle, in radians.
 * @param[out] pqr_to_ecef  Output matrix to convert from PQR to ECEF frame.
 */
OSKAR_EXPORT
void oskar_convert_pqr_to_ecef_matrix(
        const double station_ecef_coords[3],
        const double station_rotation_angle_rad,
        double pqr_to_ecef[9]
);

/**
 * @brief
 * Get conversion matrix from PQR to ENU frame.
 *
 * @details
 * Helper function for oskar_convert_pqr_to_ecef_matrix().
 * This just corresponds to a rotation around the zenith by the given angle.
 *
 * The output matrix is in row-major ("C") order.
 *
 * @param[in] station_rotation_angle_rad  Station rotation angle, in radians.
 * @param[out] pqr_to_enu  Output matrix to convert from PQR to ENU frame.
 */
OSKAR_EXPORT
void oskar_convert_pqr_to_enu_matrix(
        const double station_rotation_angle_rad,
        double pqr_to_enu[9]
);

/**
 * @brief
 * Get normal vector to Earth ellipsoid at a station position.
 *
 * @details
 * Helper function for oskar_convert_pqr_to_ecef_matrix().
 *
 * @param[in] station_ecef_coords  ECEF coordinates of station centre.
 * @param[out] norm_vec_ellipsoid  Normal vector to ellipsoid at position.
 */
OSKAR_EXPORT
void oskar_convert_ecef_to_ellipsoid_normal(
        const double station_ecef_coords[3],
        double norm_vec_ellipsoid[3]
);

/**
 * @brief
 * Get conversion matrix from ENU to ECEF frame.
 *
 * @details
 * Helper function for oskar_convert_pqr_to_ecef_matrix().
 * The equivalent function in lofarantpos is called "projection_matrix".
 *
 * The output matrix is in row-major ("C") order.
 *
 * @param[in] station_ecef_coords  ECEF coordinates of station centre.
 * @param[in] norm_vec_ellipsoid   Normal vector to ellipsoid at position.
 * @param[out] enu_to_ecef  Output matrix to convert from ENU to ECEF frame.
 */
OSKAR_EXPORT
void oskar_convert_enu_to_ecef_matrix(
        const double station_ecef_coords[3],
        const double norm_vec_ellipsoid[3],
        double enu_to_ecef[9]
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
