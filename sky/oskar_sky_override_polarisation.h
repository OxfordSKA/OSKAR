/*
 * Copyright (c) 2014, The University of Oxford
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

#ifndef OSKAR_SKY_OVERRIDE_POLARISATION_H_
#define OSKAR_SKY_OVERRIDE_POLARISATION_H_

/**
 * @file oskar_sky_override_polarisation.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Overrides source polarisation data in a sky model.
 *
 * @details
 * This function overrides source polarisation values in a sky model based
 * on the supplied mean and standard deviation of the polarisation
 * fraction and polarisation angle.
 *
 * A negative value for the mean polarisation fraction will disable the
 * function and leave the sky model unchanged.
 *
 * Stokes I values must already exist in the sky model.
 * Only the Stokes Q and U values will be modified.
 *
 * @param[out] sky               Pointer to sky model.
 * @param[in] mean_pol_fraction  Mean source polarisation fraction (range 0-1).
 * @param[in] std_pol_fraction   Standard deviation of source polarisation (range 0-1).
 * @param[in] mean_pol_angle_rad Mean source polarisation angle in radians.
 * @param[in] std_pol_angle_rad  Standard deviation of source polarisation angle in radians.
 * @param[in] seed               Random generator seed.
 * @param[in,out] status         Status return code.
 */
OSKAR_EXPORT
void oskar_sky_override_polarisation(oskar_Sky* sky, double mean_pol_fraction,
        double std_pol_fraction, double mean_pol_angle_rad,
        double std_pol_angle_rad, int seed, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SKY_OVERRIDE_POLARISATION_H_ */
