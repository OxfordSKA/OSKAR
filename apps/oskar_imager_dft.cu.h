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

#ifndef OSKAR_IMAGER_DFT_H_
#define OSKAR_IMAGER_DFT_H_

/**
 * @file oskar_imager_dft.cu.h
 */

#include "oskar_windows.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 *
 * @details
 *
 * @param[in]  num_vis      Number of visibility values.
 * @param[in]  vis          Complex visibility amplitude.
 * @param[in]  u            Baseline u co-ordinate, in metres!
 * @param[in]  v            Baseline v co-ordinate, in metres!
 * @param[in]  frequency    Frequency, in Hz.
 * @param[in]  image_size   Number of image pixels along each dimension.
 * @param[in]  l            Image l coordinates.
 * @param[out] image        Image array, must be pre allocated to a size of
 *                          num_pixels * num_pixels.
 *
 * @return CUDA error code.
 */
DllExport
int oskar_imager_dft_d(const unsigned num_vis, const double2* vis, double* u,
        double* v, const double frequency, const unsigned image_size,
        const double* l, double* image);


#ifdef __cplusplus
}
#endif

#endif // OSKAR_IMAGER_DFT_H_
