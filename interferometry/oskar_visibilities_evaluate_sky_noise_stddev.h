/*
 * Copyright (c) 2011, The University of Oxford
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


#ifndef OSKAR_VISIBILITIES_EVALUATE_SKY_NOISE_STDDEV_H_
#define OSKAR_VISIBILITIES_EVALUATE_SKY_NOISE_STDDEV_H_

/**
 * @file oskar_visibilities_evaluate_sky_noise_stddev.h
 */

#include "oskar_global.h"
#include "interferometry/oskar_Visibilities.h"
#include "interferometry/oskar_TelescopeModel.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Evaluate the standard deviation of a random gaussian visibility
 * corresponding to sky noise in Janskys.
 *
 * @details
 *
 * === WARNING ===
 * This function may add un-physical noise levels. Use with care!
 * === WARNING ===
 *
 * Current noise model is taken from the VLA science memo number 146.
 * This is expected to work for frequencies less than 408MHz although
 * this noise model should be considered in testing.
 *
 * @param[in/out] vis            visibility structure to which to add noise.
 * @param[in]     telescope      telescope model (used for evaluation of
 *                               effective area)
 * @param[in]     spectral_index Spectral index use for frequency variation
 *                               (use +0.75).
 *
 * @return An error code.
 */
OSKAR_EXPORT
int oskar_visibilities_evaluate_sky_noise_stddev(oskar_Visibilities* vis,
        const oskar_TelescopeModel* telescope, double spectral_index);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_VISIBILITIES_EVALUATE_SKY_NOISE_PARAMETERS_H_ */
