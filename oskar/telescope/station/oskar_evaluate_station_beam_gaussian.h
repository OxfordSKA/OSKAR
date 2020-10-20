/*
 * Copyright (c) 2012-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_EVALUATE_STATION_BEAM_GAUSSIAN_H_
#define OSKAR_EVALUATE_STATION_BEAM_GAUSSIAN_H_

/**
 * @file oskar_evaluate_station_beam_gaussian.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates a Gaussian station beam.
 *
 * @details
 * This function evaluates scalar station beam amplitudes in the form of
 * a circular Gaussian, specified by its full width at half maximum,
 * \p fwhm_deg.
 *
 * @param[in] num_points   Number of points at which to evaluate beam.
 * @param[in] l            Beam l-direction cosines.
 * @param[in] m            Beam m-direction cosines.
 * @param[in] horizon_mask Positions with mask values < 0 are zeroed in output.
 * @param[in] fwhm_rad     Gaussian FWHM of beam, in degrees.
 * @param[out] beam        Output beam values.
 * @param[in,out] status   Status return code.
 */
OSKAR_EXPORT
void oskar_evaluate_station_beam_gaussian(int num_points,
        const oskar_Mem* l, const oskar_Mem* m, const oskar_Mem* horizon_mask,
        double fwhm_rad, oskar_Mem* beam, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
