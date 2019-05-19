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

#ifndef OSKAR_APPLY_ELEMENT_TAPER_COSINE_H_
#define OSKAR_APPLY_ELEMENT_TAPER_COSINE_H_

/**
 * @file oskar_apply_element_taper_cosine.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Function to apply a cosine taper to the element response.
 *
 * @details
 * This function multiplies the response of the element by a
 * cosine taper. The multiplication is performed in-place.
 *
 * @param[in] num_sources  Number of source positions.
 * @param[in] cos_power    Power of cosine(theta) function.
 * @param[in] theta        Array of source theta values, in radians.
 * @param[in] offset_out   Start offset into output array.
 * @param[in,out] jones    Array of Jones matrices.
 * @param[in,out] status   Status return code.
 */
OSKAR_EXPORT
void oskar_apply_element_taper_cosine(int num_sources, double cos_power,
        const oskar_Mem* theta, int offset_out, oskar_Mem* jones, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_APPLY_ELEMENT_TAPER_COSINE_H_ */
