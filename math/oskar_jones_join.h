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

#ifndef OSKAR_JONES_JOIN_H_
#define OSKAR_JONES_JOIN_H_

/**
 * @file oskar_jones_join.h
 */

#include "oskar_global.h"
#include "math/oskar_Jones.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Wrapper function to join two Jones matrices.
 *
 * @details
 * This function multiplies together a set of two Jones matrices.
 *
 * Matrix multiplication is done in the order J3 = J1 * J2.
 *
 * If the parameter \p j3 is NULL on input, then J3 = J1 instead, and the
 * multiplication is performed in-place.
 *
 * The input dimensions (number of sources, number of stations) must be the
 * same for J3, J1 and J2, and the data type (single precision or double
 * precision) must also be consistent.
 *
 * The element size of J3 should be greater than or equal to the element
 * size of J2. For example, J3 could be a full 2x2 complex matrix and J2 a
 * complex scalar, but not vice versa.
 *
 * @param[in,out] j3 If not NULL, then pointer to the output data structure.
 * @param[in,out] j1 On input, pointer to data structure for the first set of
 *                   matrices; on output, the result, if \p j3 is NULL.
 * @param[in]     j2 Pointer to data structure for the second set of matrices.
 *
 * @return
 * This function returns a code to indicate if there were errors in execution:
 * - A return code of 0 indicates no error.
 * - A return code of -1 indicates that the memory in J1 is unallocated.
 * - A return code of -2 indicates that the memory in J2 is unallocated.
 * - A return code of -3 indicates that the memory in J3 is unallocated.
 * - A return code of -11 indicates that the input matrix blocks have
 *   different source dimensions.
 * - A return code of -12 indicates that the input matrix blocks have
 *   different station dimensions.
 * - A return code of -20 indicates that the element size of J3 is smaller than
 *   the element size of J2 or J1.
 * - A return code of -100 indicates a type mismatch between J1 and J2.
 */
OSKAR_EXPORT
int oskar_jones_join(oskar_Jones* j3, oskar_Jones* j1, const oskar_Jones* j2);

#ifdef __cplusplus
}
#endif

#endif // OSKAR_JONES_JOIN_H_
