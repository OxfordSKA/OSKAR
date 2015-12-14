/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#ifndef OSKAR_IMAGE_RESIZE_H_
#define OSKAR_IMAGE_RESIZE_H_

/**
 * @file oskar_image_resize.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * NOTE consider reordering function arguments with slowest varying dimension
 * first. as this is more conventional with multi-dimensional array
 * dereferencing.
 * i.e. array[m][n] would have m as the slowest varying dimension.
 */
/**
 * @brief
 * Resizes memory held by an image structure.
 *
 * @details
 * This function resizes memory held by an image structure.
 *
 * @param[in] image          Pointer to image structure.
 * @param[in] width          Required width of image in pixels.
 * @param[in] height         Required height of image in pixels.
 * @param[in] num_pols       Required size of polarisation dimension.
 * @param[in] num_times      Required size of time dimension.
 * @param[in] num_channels   Required size of frequency dimension.
 * @param[in,out] status     Status return code.
 */
OSKAR_EXPORT
void oskar_image_resize(oskar_Image* image, int width, int height,
        int num_pols, int num_times, int num_channels, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_IMAGE_RESIZE_H_ */
