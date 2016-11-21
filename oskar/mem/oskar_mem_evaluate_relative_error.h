/*
 * Copyright (c) 2013, The University of Oxford
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

#ifndef OSKAR_MEM_EVALUATE_RELATIVE_ERROR_H_
#define OSKAR_MEM_EVALUATE_RELATIVE_ERROR_H_

/**
 * @file oskar_mem_evaluate_relative_error.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Returns the minimum, maximum, average and standard deviation of the
 * relative error between elements of two floating-point arrays.
 *
 * @details
 * This function computes statistics to return the minimum, maximum,
 * average and population standard deviation of the relative error between
 * elements of two floating-point arrays.
 *
 * Data in GPU memory are copied back to the host first, if necessary.
 *
 * The relative error is computed using abs(approx / accurate - 1).
 * When using complex data types, the relative error is computed using the
 * complex magnitude rather than the individual real and imaginary components.
 *
 * See "Error Bounds on Complex Floating-Point Multiplication"
 * by Brent, Percival and Zimmerman, in "Mathematics of Computation" (AMS),
 * Volume 76, Number 259, Page 1469.
 * Online at http://www.jstor.org/stable/40234438
 *
 * This function is intended to check values within memory blocks for
 * consistency in unit tests.
 *
 * @param[in] val_approx     Array of approximate (lower precision) data.
 * @param[in] val_accurate   Array of accurate (higher precision) data.
 * @param[out] min_rel_error Minimum relative error.
 * @param[out] max_rel_error Maximum relative error.
 * @param[out] avg_rel_error Mean relative error.
 * @param[out] std_rel_error Population standard deviation of relative error.
 * @param[in,out] status     Status return code.
 */
OSKAR_EXPORT
void oskar_mem_evaluate_relative_error(const oskar_Mem* val_approx,
        const oskar_Mem* val_accurate, double* min_rel_error,
        double* max_rel_error, double* avg_rel_error, double* std_rel_error,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_MEM_EVALUATE_RELATIVE_ERROR_H_ */
