/*
 * Copyright (c) 2015, The University of Oxford
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

#ifndef OSKAR_VIS_HEADER_CREATE_H_
#define OSKAR_VIS_HEADER_CREATE_H_

/**
 * @file oskar_vis_header_create.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Creates a new visibility header structure.
 *
 * @details
 * This function creates a new visibility header structure in memory and
 * returns a handle to it.
 *
 * Allowed values of the \p amp_type parameter are
 * - OSKAR_SINGLE_COMPLEX
 * - OSKAR_DOUBLE_COMPLEX
 * - OSKAR_SINGLE_COMPLEX_MATRIX
 * - OSKAR_DOUBLE_COMPLEX_MATRIX
 *
 * The structure must be deallocated using oskar_vis_header_free() when it is
 * no longer required.
 *
 * @param[in] amp_type               OSKAR memory type of visibility amplitudes.
 * @param[in] coord_precision        OSKAR memory type of coordinates.
 * @param[in] max_times_per_block    Maximum number of time samples per block.
 * @param[in] num_times_total        Number of time samples in total.
 * @param[in] max_channels_per_block Maximum number of channels per block.
 * @param[in] num_channels_total     Number of channels in total.
 * @param[in] num_stations           Number of stations.
 * @param[in] write_autocorr         Set if auto-correlations must be written.
 * @param[in] write_crosscorr        Set if cross-correlations must be written.
 * @param[in,out]  status            Status return code.
 *
 * @return A handle to the new data structure.
 */
OSKAR_EXPORT
oskar_VisHeader* oskar_vis_header_create(int amp_type, int coord_precision,
        int max_times_per_block, int num_times_total,
        int max_channels_per_block, int num_channels_total, int num_stations,
        int write_autocorr, int write_crossscor, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_VIS_HEADER_CREATE_H_ */
