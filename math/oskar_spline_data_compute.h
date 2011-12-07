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

#ifndef OSKAR_SPLINE_DATA_COMPUTE_H_
#define OSKAR_SPLINE_DATA_COMPUTE_H_

/**
 * @file oskar_spline_data_compute.h
 */

#include "oskar_global.h"
#include "math/oskar_SplineData.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Computes spline data from a table.
 *
 * @details
 * This function computes all required spline data from an input data table.
 *
 * Note that the fastest varying dimension is along y.
 *
 * @param[in,out] spline Pointer to data structure.
 * @param[in] num_x The number of input grid points in the x dimension.
 * @param[in] num_y The number of input grid points in the y dimension.
 * @param[in] start_x The value of x on the first grid line.
 * @param[in] start_y The value of y on the first grid line.
 * @param[in] end_x The value of x on the last grid line.
 * @param[in] end_y The value of y on the last grid line.
 * @param[in] data Pointer to look-up table data.
 *
 * @return
 * This function returns a code to indicate if there were errors in execution:
 * - A return code of 0 indicates no error.
 * - A positive return code indicates a CUDA error.
 * - A negative return code indicates an OSKAR error.
 */
OSKAR_EXPORT
int oskar_spline_data_compute(oskar_SplineData* spline, int num_x, int num_y,
        double start_x, double start_y, double end_x, double end_y,
        const oskar_Mem* data);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SPLINE_DATA_COMPUTE_H_ */
