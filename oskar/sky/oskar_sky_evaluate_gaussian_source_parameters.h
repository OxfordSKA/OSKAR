/*
 * Copyright (c) 2012-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_EVALUATE_GAUSSIAN_SOURCE_PARAMETERS_H_
#define OSKAR_SKY_EVALUATE_GAUSSIAN_SOURCE_PARAMETERS_H_

/**
 * @file oskar_sky_evaluate_gaussian_source_parameters.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Evaluates Gaussian parameters for extended sources.
 *
 * @details
 * This is done by projecting the source ellipse as defined on the sky
 * to the observation l,m plane.
 *
 * - Six points are evaluated on the circumference of the ellipse which defines
 *   the gaussian source
 * - These points are projected to the l,m plane
 * - Points on the l,m plane are then used to fit a new ellipse which defines
 *   the l,m plane gaussian function of the source
 * - 2D Gaussian parameters are evaluated from the fit of the l,m plane ellipse.
 *
 * Fitting of the ellipse on the l,m plane is carried out by oskar_fit_ellipse()
 * which uses the LAPACK routines (D|S)GETRS and (D|S)GETRF to perform the
 * fitting.
 *
 * @param[in,out] sky      Sky model to update.
 * @param[in] zero_failed_sources If set, zero amplitude of sources
 *                                where the Gaussian solution fails.
 * @param[in] ra0_rad      Right ascension of the phase centre, in radians.
 * @param[in] dec0_rad     Declination of the phase centre, in radians.
 * @param[out] num_failed  The number of sources where Gaussian fitting failed.
 * @param[in,out] status   Status return code.
 */
OSKAR_EXPORT
void oskar_sky_evaluate_gaussian_source_parameters(
        oskar_Sky* sky,
        int zero_failed_sources,
        double ra0_rad,
        double dec0_rad,
        int* num_failed,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SKY_EVALUATE_GAUSSIAN_SOURCE_PARAMETERS_H_ */
