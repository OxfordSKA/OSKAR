/*
 * Copyright (c) 2019-2020, The University of Oxford
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

#ifndef OSKAR_TELESCOPE_OVERRIDE_ELEMENT_GAINS_H_
#define OSKAR_TELESCOPE_OVERRIDE_ELEMENT_GAINS_H_

/**
 * @file oskar_telescope_override_element_gains.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Overrides element gains for all stations in a telescope model.
 *
 * @details
 * This function overrides element gains for all stations in a telescope model
 * using the supplied mean and standard deviation.
 *
 * Only the gains at the deepest (element) level are set.
 *
 * The telescope model must be in CPU-accessible memory.
 *
 * @param[in,out] t          Telescope model to modify.
 * @param[in] feed           Feed index (0 = X, 1 = Y).
 * @param[in] seed           Random generator seed.
 * @param[in] gain_mean      Mean element gain.
 * @param[in] gain_std       Standard deviation of element gain.
 * @param[in,out] status     Status return code.
 */
OSKAR_EXPORT
void oskar_telescope_override_element_gains(oskar_Telescope* t, int feed,
        unsigned int seed, double gain_mean, double gain_std, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
