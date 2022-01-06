/*
 * Copyright (c) 2012-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_EVALUATE_JONES_R_H_
#define OSKAR_EVALUATE_JONES_R_H_

/**
 * @file oskar_evaluate_jones_R.h
 */

#include <oskar_global.h>
#include <sky/oskar_sky.h>
#include <telescope/oskar_telescope.h>
#include <interferometer/oskar_jones.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Function to construct matrices for parallactic angle rotation and
 * conversion of linear Stokes parameters from equatorial to local horizontal
 * frame.
 *
 * @details
 * This function constructs a set of Jones matrices that will transform the
 * equatorial Stokes parameters into the local horizontal frame of each station.
 * This corresponds to a rotation by the parallactic angle (q) for each source
 * and station. The Jones matrix is:
 *
 * ( cos(q)  -sin(q) )
 * ( sin(q)   cos(q) )
 *
 * @param[out] R          Output set of Jones matrices.
 * @param[in] num_sources Number of sources to use from coordinate arrays.
 * @param[in] ra_rad      Input Right Ascension values, in radians.
 * @param[in] dec_rad     Input Declination values, in radians.
 * @param[in] telescope   Input telescope model.
 * @param[in] gast        The Greenwich Apparent Sidereal Time, in radians.
 * @param[in,out] status  Status return code.
 */
OSKAR_EXPORT
void oskar_evaluate_jones_R(
        oskar_Jones* R,
        int num_sources,
        const oskar_Mem* ra_rad,
        const oskar_Mem* dec_rad,
        const oskar_Telescope* telescope,
        double gast,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
