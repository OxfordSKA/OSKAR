/*
 * Copyright (c) 2016-2017, The University of Oxford
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

#ifndef OSKAR_IMAGER_RUN_H_
#define OSKAR_IMAGER_RUN_H_

/**
 * @file oskar_imager_run.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * High-level function to make images from visibility data in files.
 *
 * @details
 * After creating and setting up the imager, this function is used to make
 * one or more images from visibility data in OSKAR visibility files or
 * (if compiled with casacore) CASA Measurement Sets.
 *
 * Call oskar_imager_set_input_files() to set the filenames.
 * When using Measurement Sets, the filename must end with ".ms" or ".MS".
 *
 * Copies of the image and/or grid planes can be returned if required
 * by supplying arrays as input arguments. Set these to NULL if not required.
 *
 * @param[in,out] h             Handle to imager.
 * @param[in] num_output_images Number of output image planes supplied.
 * @param[in] output_images     Array of image planes.
 * @param[in] num_output_grids  Number of output grid planes supplied.
 * @param[in] output_grids      Array of grid planes.
 * @param[in,out] status        Status return code.
 */
OSKAR_EXPORT
void oskar_imager_run(oskar_Imager* h,
        int num_output_images, oskar_Mem** output_images,
        int num_output_grids, oskar_Mem** output_grids, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_IMAGER_RUN_H_ */
