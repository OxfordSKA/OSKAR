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


#ifndef OSKAR_EVALUATE_SKY_TEMPERATURE_H_
#define OSKAR_EVALUATE_SKY_TEMPERATURE_H_

/**
 * @file oskar_evaluate_sky_temperature.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates the sky temperature (in Kelvin).
 *
 * @details
 * See VLA memo 146.
 *
 * Sensible defaults:
 *  - \p freq0: 408.0e6
 *  - \p temp0: 20.0
 *  - \p spectral_index: 0.75.
 *
 * @param[out] temp           The sky temperature, in K.
 * @param[in]  freq           Observation frequency, in Hz.
 * @param[in]  freq0          Reference frequency, in Hz.
 * @param[in]  temp0          Temperature at the reference frequency, in K.
 * @param[in]  spectral_index Spectral index of the frequency.
 *
 * @return An error code.
 */
OSKAR_EXPORT
int oskar_evaluate_sky_temperature(double* temp, double freq,
        double freq0, double temp0, double spectral_index);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_SKY_TEMPERATURE_H_ */
