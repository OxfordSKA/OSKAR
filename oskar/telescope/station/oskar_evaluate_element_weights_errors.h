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

#ifndef OSKAR_EVALUATE_ELEMENT_WEIGHTS_ERRORS_H_
#define OSKAR_EVALUATE_ELEMENT_WEIGHTS_ERRORS_H_

/**
 * @file oskar_evaluate_element_weights_errors.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates weights errors for antenna elements.
 *
 * @details
 * Weights errors are generated randomly from a Gaussian distribution.
 *
 * @param[in] num_elements The number of antenna elements.
 * @param[in] gain         Element gain values.
 * @param[in] gain_error   Standard deviation for time-variable element gains.
 * @param[in] phase        Element phase offsets, in radians.
 * @param[in] phase_error  Standard deviation for time-variable element phases.
 * @param[in] random_seed  Random seed.
 * @param[in] time_index   The simulation time index.
 * @param[in] station_id   Station ID. Must be unique for the given time.
 * @param[out] errors      Complex element errors for this time.
 * @param[in,out] status   Status return code.
 */
OSKAR_EXPORT
void oskar_evaluate_element_weights_errors(int num_elements,
        const oskar_Mem* gain, const oskar_Mem* gain_error,
        const oskar_Mem* phase, const oskar_Mem* phase_error,
        unsigned int random_seed, int time_index, int station_id,
        oskar_Mem* errors, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_ELEMENT_WEIGHTS_ERRORS_H_ */
