/*
 * Copyright (c) 2012, The University of Oxford
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


#ifndef OSKAR_EVALUATE_IMAGE_LMN_POINT_H_
#define OSKAR_EVALUATE_IMAGE_LMN_POINT_H_

/**
 * @file oskar_evaluate_image_lm_point.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * TODO name? oskar_evaluate_relative_lnm() might be better?
 *
 * @brief Evaluates the @p l, @p m, @p n coordinates of a point at @p ra,
 * @p dec relative to an image phase centre at @p ra0,  @p dec0
 *
 * @details
 * Used for phase rotation of visibilities.
 *
 * @param l     Direction cosine
 * @param m     Direction cosine
 * @param n     Direction cosine
 * @param ra0   Phase reference position, in radians.
 * @param dec0  Phase reference position, in radians.
 * @param ra    Phase direction, in radians.
 * @param dec   Phase direction, in radians.
 */
OSKAR_EXPORT
void oskar_evaluate_image_lmn_point(double*l, double* m, double* n,
        double ra0, double dec0, double ra, double dec);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_IMAGE_LMN_POINT_H_ */
