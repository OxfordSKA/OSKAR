/*
 * Copyright (c) 2012-2019, The University of Oxford
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

#ifndef OSKAR_SKY_SCALE_FLUX_WITH_FREQUENCY_H_
#define OSKAR_SKY_SCALE_FLUX_WITH_FREQUENCY_H_

/**
 * @file oskar_sky_scale_flux_with_frequency.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Scales all current source brightnesses according to the spectral index for
 * the given frequency.
 *
 * @details
 * This function evaluates all fluxes (all Stokes parameters) at the specified
 * frequency using the spectral index of each source. The reference frequency
 * of each source is also updated to the specified frequency.
 *
 * Frequency scaling is performed using the expression:
 *
 * \f[
 * F = F * (\nu / \nu_0)^\alpha
 * \f]
 *
 * where \f$F\f$ is the flux, \f$\nu\f$ is the new frequency, \f$\nu_0\f$ is
 * the reference frequency, and \f$\alpha\f$ is the spectral index value.
 *
 * @param[in,out] sky The sky model to re-scale.
 * @param[in] frequency The required frequency, in Hz.
 * @param[in,out] status   Status return code.
 */
OSKAR_EXPORT
void oskar_sky_scale_flux_with_frequency(oskar_Sky* sky, double frequency,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SKY_SCALE_FLUX_WITH_FREQUENCY_H_ */
