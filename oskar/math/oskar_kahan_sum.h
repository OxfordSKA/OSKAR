/*
 * Copyright (c) 2013-2018, The University of Oxford
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

#ifndef OSKAR_KAHAN_SUM_H_
#define OSKAR_KAHAN_SUM_H_

/**
 * @file oskar_kahan_sum.h
 */

#include <oskar_global.h>
#include <utility/oskar_vector_types.h>

/**
 * @brief
 * Performs Kahan summation.
 *
 * @details
 * Performs Kahan summation to avoid loss of precision.
 *
 * @param[in]     REAL    Either 'float' or 'double'.
 * @param[in,out] SUM     Updated sum.
 * @param[in]     VAL     Value to add to sum.
 * @param[in,out] GUARD   Guard value (initially 0; preserve between calls).
 */
#define OSKAR_KAHAN_SUM(REAL, SUM, VAL, GUARD) {                          \
        const REAL y__ = VAL - GUARD;                                     \
        const REAL t__ = SUM + y__;                                       \
        GUARD = (t__ - SUM) - y__;                                        \
        SUM = t__; }

/**
 * @brief
 * Performs Kahan summation of a complex number.
 *
 * @details
 * Performs Kahan summation to avoid loss of precision.
 *
 * @param[in]     REAL    Either 'float' or 'double'.
 * @param[in,out] SUM     Updated sum.
 * @param[in]     VAL     Value to add to sum.
 * @param[in,out] GUARD   Guard value (initially 0; preserve between calls).
 */
#define OSKAR_KAHAN_SUM_COMPLEX(REAL, SUM, VAL, GUARD) {                  \
        OSKAR_KAHAN_SUM(REAL, SUM.x, VAL.x, GUARD.x);                     \
        OSKAR_KAHAN_SUM(REAL, SUM.y, VAL.y, GUARD.y); }

/**
 * @brief
 * Performs Kahan multiply-add of a complex number.
 *
 * @details
 * Performs Kahan summation to avoid loss of precision.
 *
 * @param[in]     REAL    Either 'float' or 'double'.
 * @param[in,out] SUM     Updated sum.
 * @param[in]     VAL     Value to add to sum.
 * @param[in]     F       Factor by which to multiply input value before summation.
 * @param[in,out] GUARD   Guard value (initially 0; preserve between calls).
 */
#define OSKAR_KAHAN_SUM_MULTIPLY_COMPLEX(REAL, SUM, VAL, F, GUARD) {      \
        OSKAR_KAHAN_SUM(REAL, SUM.x, (VAL.x * F), GUARD.x);               \
        OSKAR_KAHAN_SUM(REAL, SUM.y, (VAL.y * F), GUARD.y); }

/**
 * @brief
 * Performs Kahan summation of a complex matrix.
 *
 * @details
 * Performs Kahan summation to avoid loss of precision.
 *
 * @param[in]     REAL    Either 'float' or 'double'.
 * @param[in,out] SUM     Updated sum.
 * @param[in]     VAL     Value to add to sum.
 * @param[in,out] GUARD   Guard value (initially 0; preserve between calls).
 */
#define OSKAR_KAHAN_SUM_COMPLEX_MATRIX(REAL, SUM, VAL, GUARD) {           \
        OSKAR_KAHAN_SUM_COMPLEX(REAL, SUM.a, VAL.a, GUARD.a);             \
        OSKAR_KAHAN_SUM_COMPLEX(REAL, SUM.b, VAL.b, GUARD.b);             \
        OSKAR_KAHAN_SUM_COMPLEX(REAL, SUM.c, VAL.c, GUARD.c);             \
        OSKAR_KAHAN_SUM_COMPLEX(REAL, SUM.d, VAL.d, GUARD.d); }

/**
 * @brief
 * Performs Kahan multiply-add of a complex matrix.
 *
 * @details
 * Performs Kahan summation to avoid loss of precision.
 *
 * @param[in]     REAL    Either 'float' or 'double'.
 * @param[in,out] SUM     Updated sum.
 * @param[in]     VAL     Value to add to sum.
 * @param[in]     F       Factor by which to multiply input value before summation.
 * @param[in,out] GUARD   Guard value (initially 0; preserve between calls).
 */
#define OSKAR_KAHAN_SUM_MULTIPLY_COMPLEX_MATRIX(REAL, SUM, VAL, F, GUARD) {\
        OSKAR_KAHAN_SUM_MULTIPLY_COMPLEX(REAL, SUM.a, VAL.a, F, GUARD.a); \
        OSKAR_KAHAN_SUM_MULTIPLY_COMPLEX(REAL, SUM.b, VAL.b, F, GUARD.b); \
        OSKAR_KAHAN_SUM_MULTIPLY_COMPLEX(REAL, SUM.c, VAL.c, F, GUARD.c); \
        OSKAR_KAHAN_SUM_MULTIPLY_COMPLEX(REAL, SUM.d, VAL.d, F, GUARD.d); }

#endif /* OSKAR_KAHAN_SUM_H_ */
