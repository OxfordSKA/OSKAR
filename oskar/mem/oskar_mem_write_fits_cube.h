/*
 * Copyright (c) 2017, The University of Oxford
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

#ifndef OSKAR_MEM_WRITE_FITS_CUBE_H_
#define OSKAR_MEM_WRITE_FITS_CUBE_H_

/**
 * @file oskar_mem_write_fits_cube.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Writes pixel data to a FITS image file.
 *
 * @details
 * Writes pixel data to a FITS image file.
 *
 * Complex data are written with the real and imaginary parts going to
 * the first and second HDUs, respectively.
 *
 * Can be called multiple times with the same \p root_name to write
 * one plane at a time.
 *
 * Set \p i_plane to -1 if \p data contains the entire cube.
 *
 * @param[in] data                Array of values to write.
 * @param[in] root_name           Root name of FITS file to write.
 * @param[in] width               Image width, in pixels.
 * @param[in] height              Image height, in pixels.
 * @param[in] num_planes          Number of image planes in the file.
 * @param[in] i_plane             Image plane index to write (-1 = all).
 * @param[in,out] status          Status return code.
 */
OSKAR_EXPORT
void oskar_mem_write_fits_cube(oskar_Mem* data, const char* root_name,
        int width, int height, int num_planes, int i_plane, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_MEM_WRITE_FITS_CUBE_H_ */
