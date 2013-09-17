/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#ifndef OSKAR_EVALUATE_STATION_BEAM_GAUSSIAN_H_
#define OSKAR_EVALUATE_STATION_BEAM_GAUSSIAN_H_

/**
 * @file oskar_evaluate_station_beam_gaussian.h
 */

#include <oskar_global.h>
#include <oskar_mem.h>

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
 * @param[out] beam        Output beam values.
 * @param[in] num_points   Number of points at which to evaluate beam.
 * @param[in] l            Beam l-direction cosines.
 * @param[in] m            Beam m-direction cosines.
 * @param[in] horizon_mask Positions with mask values < 0 are zeroed in output.
 * @param[in] fwhm_rad     Gaussian FWHM of beam, in degrees.
 * @param[in,out] status   Status return code.
 */
OSKAR_EXPORT
void oskar_evaluate_station_beam_gaussian(oskar_Mem* beam,
        int num_points, const oskar_Mem* l, const oskar_Mem* m,
        const oskar_Mem* horizon_mask, double fwhm_rad, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_STATION_BEAM_GAUSSIAN_H_ */
