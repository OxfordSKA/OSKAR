/*
 * Copyright (c) 2011-2014, The University of Oxford
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

#ifndef OSKAR_MAKE_IMAGE_DFT_H_
#define OSKAR_MAKE_IMAGE_DFT_H_

/**
 * @file oskar_make_image_dft.h
 */

#include <oskar_global.h>
#include <oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Makes an image of a set of visibilities using a DFT.
 *
 * @details
 * This function uses a DFT (on the GPU) to make an image of a set of
 * visibilities. The arrays of direction cosines must both be the same
 * length, and correspond to the required pixel positions in the final
 * image (they do not necessarily need to be on a grid).
 *
 * The output image is purely real and the pixel order is the same as the
 * input l,m positions.
 *
 * @param[out] image        Output image.
 * @param[in]  uu_metres    Baseline u co-ordinates, in metres.
 * @param[in]  vv_metres    Baseline v co-ordinates, in metres.
 * @param[in]  amp          Complex visibility amplitude.
 * @param[in]  l            Array of image l coordinates.
 * @param[in]  m            Array of image m coordinates.
 * @param[in]  frequency_hz Frequency, in Hz.
 * @param[in,out] status    Status return code.
 */
OSKAR_EXPORT
void oskar_make_image_dft(oskar_Mem* image, const oskar_Mem* uu_metres,
        const oskar_Mem* vv_metres, const oskar_Mem* amp, const oskar_Mem* l,
        const oskar_Mem* m, double frequency_hz, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_MAKE_IMAGE_DFT_H_ */
