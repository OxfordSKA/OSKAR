/*
 * Copyright (c) 2016, The University of Oxford
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

#ifndef OSKAR_GRID_FUNCTIONS_PILLBOX_H_
#define OSKAR_GRID_FUNCTIONS_PILLBOX_H_

/**
 * @file oskar_grid_functions_pillbox.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Generates pillbox grid convolution function (GCF).
 *
 * @details
 * Generates pillbox grid convolution function (GCF) consistent with CASA.
 *
 * @param[in] support    GCF support size (typ. 3; width = 2 * support + 1).
 * @param[in] oversample GCF oversample factor, or values per grid cell.
 * @param[in,out] fn     GCF array, length oversample * (support + 1).
 */
OSKAR_EXPORT
void oskar_grid_convolution_function_pillbox(const int support,
        const int oversample, double* fn);

/**
 * Generates grid correction function for pillbox convolution function.
 *
 * @param[in] image_size Side length of image.
 * @param[in,out] fn     Array holding correction function, length image_size.
 */
OSKAR_EXPORT
void oskar_grid_correction_function_pillbox(const int image_size,
        double* fn);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_GRID_FUNCTIONS_PILLBOX_H_ */
