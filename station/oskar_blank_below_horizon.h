/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#ifndef OSKAR_BLANK_BELOW_HORIZON_H_
#define OSKAR_BLANK_BELOW_HORIZON_H_

/**
 * @file oskar_blank_below_horizon.h
 */

#include <oskar_global.h>
#include <oskar_mem.h>
#include <oskar_vector_types.h>

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_EXPORT
void oskar_blank_below_horizon_matrix_f(float4c* jones, int num_sources,
        const float* mask);

OSKAR_EXPORT
void oskar_blank_below_horizon_scalar_f(float2* jones, int num_sources,
        const float* mask);

OSKAR_EXPORT
void oskar_blank_below_horizon_matrix_d(double4c* jones, int num_sources,
        const double* mask);

OSKAR_EXPORT
void oskar_blank_below_horizon_scalar_d(double2* jones, int num_sources,
        const double* mask);

/**
 * @brief
 * Function to blank sources below the horizon
 *
 * @details
 * This function sets individual Jones matrices or scalars to zero for those
 * sources that are below the horizon.
 *
 * For sources where the mask value is negative, the corresponding element
 * of the Jones data array is set to zero.
 *
 * @param[in,out] data    Array of Jones matrices or scalars per source.
 * @param[in] mask        Array of mask values.
 * @param[in] num_sources Number of sources in arrays.
 * @param[in,out] status  Status return code.
 */
OSKAR_EXPORT
void oskar_blank_below_horizon(oskar_Mem* data, const oskar_Mem* mask,
        int num_sources, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_BLANK_BELOW_HORIZON_H_ */
