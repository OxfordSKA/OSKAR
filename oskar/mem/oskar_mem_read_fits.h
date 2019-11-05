/*
 * Copyright (c) 2019, The University of Oxford
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

#ifndef OSKAR_MEM_READ_FITS_H_
#define OSKAR_MEM_READ_FITS_H_

/**
 * @file oskar_mem_read_fits.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Read pixel data from a FITS image file.
 *
 * @details
 * Read pixel data from a FITS image file.
 *
 * If \p num_dims is 0, return only the axis dimensions.
 * If \p num_dims is -1, read the entire cube.
 *
 * Note that \p axis_size and \p axis_inc are re-allocated internally,
 * if necessary, and must be freed by the caller.
 *
 * @param[in,out] data            Array to fill.
 * @param[in] offset              Output offset.
 * @param[in] num_pixels          Number of pixels to read.
 * @param[in] file_name           File name of FITS file to read.
 * @param[in] num_index_dims      Number of dimensions in plane index.
 * @param[in] start_index         Zero-based axis indices to read from.
 * @param[in,out] num_axes        Number of dimensions in the cube.
 * @param[in,out] axis_size       Size of each axis.
 * @param[in,out] axis_inc        Increment of each axis.
 * @param[in,out] status          Status return code.
 */
OSKAR_EXPORT
void oskar_mem_read_fits(oskar_Mem* data, size_t offset, size_t num_pixels,
        const char* file_name, int num_index_dims, const int* start_index,
        int* num_axes, int** axis_size, double** axis_inc, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
