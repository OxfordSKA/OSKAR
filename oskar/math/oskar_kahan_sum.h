/*
 * Copyright (c) 2013-2014, The University of Oxford
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

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Performs Kahan summation (single precision).
 *
 * @details
 * Performs Kahan summation to avoid loss of precision.
 *
 * @param[in,out] sum Updated sum.
 * @param[in]     val Value to add to sum.
 * @param[in,out] c   Guard value (initially 0.0; must preserve between calls).
 */
OSKAR_INLINE
void oskar_kahan_sum_f(float* sum, const float val, float* c)
{
    float y, t;
    y = val - *c;
    t = *sum + y;
    *c = (t - *sum) - y;
    *sum = t;
}

/**
 * @brief
 * Performs Kahan summation of a complex scalar (single precision).
 *
 * @details
 * Performs Kahan summation to avoid loss of precision.
 *
 * @param[in,out] sum Updated sum.
 * @param[in]     val Value to add to sum.
 * @param[in,out] c   Guard value (initially 0.0; must preserve between calls).
 */
OSKAR_INLINE
void oskar_kahan_sum_complex_f(float2* sum, const float2 val, float2* c)
{
    oskar_kahan_sum_f(&sum->x, val.x, &c->x);
    oskar_kahan_sum_f(&sum->y, val.y, &c->y);
}

/**
 * @brief
 * Performs Kahan multiply-add of a complex scalar (single precision).
 *
 * @details
 * Performs Kahan summation to avoid loss of precision.
 *
 * @param[in,out] sum Updated sum.
 * @param[in]     val Value to add to sum.
 * @param[in]     f   Factor by which to multiply input value before summation.
 * @param[in,out] c   Guard value (initially 0.0; must preserve between calls).
 */
OSKAR_INLINE
void oskar_kahan_sum_multiply_complex_f(float2* sum, const float2 val,
        const float f, float2* c)
{
    oskar_kahan_sum_f(&sum->x, val.x * f, &c->x);
    oskar_kahan_sum_f(&sum->y, val.y * f, &c->y);
}

/**
 * @brief
 * Performs Kahan summation of a complex matrix (single precision).
 *
 * @details
 * Performs Kahan summation to avoid loss of precision.
 *
 * @param[in,out] sum Updated sum.
 * @param[in]     val Value to add to sum.
 * @param[in,out] c   Guard value (initially 0.0; must preserve between calls).
 */
OSKAR_INLINE
void oskar_kahan_sum_complex_matrix_f(float4c* sum, const float4c val,
        float4c* c)
{
    oskar_kahan_sum_complex_f(&sum->a, val.a, &c->a);
    oskar_kahan_sum_complex_f(&sum->b, val.b, &c->b);
    oskar_kahan_sum_complex_f(&sum->c, val.c, &c->c);
    oskar_kahan_sum_complex_f(&sum->d, val.d, &c->d);
}

/**
 * @brief
 * Performs Kahan multiply-add of a complex matrix (single precision).
 *
 * @details
 * Performs Kahan summation to avoid loss of precision.
 *
 * @param[in,out] sum Updated sum.
 * @param[in]     val Value to add to sum.
 * @param[in]     f   Factor by which to multiply input value before summation.
 * @param[in,out] c   Guard value (initially 0.0; must preserve between calls).
 */
OSKAR_INLINE
void oskar_kahan_sum_multiply_complex_matrix_f(float4c* sum,
        const float4c val, const float f, float4c* c)
{
    oskar_kahan_sum_multiply_complex_f(&sum->a, val.a, f, &c->a);
    oskar_kahan_sum_multiply_complex_f(&sum->b, val.b, f, &c->b);
    oskar_kahan_sum_multiply_complex_f(&sum->c, val.c, f, &c->c);
    oskar_kahan_sum_multiply_complex_f(&sum->d, val.d, f, &c->d);
}

/**
 * @brief
 * Performs Kahan summation (double precision).
 *
 * @details
 * Performs Kahan summation to avoid loss of precision.
 *
 * @param[in,out] sum Updated sum.
 * @param[in]     val Value to add to sum.
 * @param[in,out] c   Guard value (initially 0.0; must preserve between calls).
 */
OSKAR_INLINE
void oskar_kahan_sum_d(double* sum, const double val, double* c)
{
    double y, t;
    y = val - *c;
    t = *sum + y;
    *c = (t - *sum) - y;
    *sum = t;
}

/**
 * @brief
 * Performs Kahan summation of a complex scalar (double precision).
 *
 * @details
 * Performs Kahan summation to avoid loss of precision.
 *
 * @param[in,out] sum Updated sum.
 * @param[in]     val Value to add to sum.
 * @param[in,out] c   Guard value (initially 0.0; must preserve between calls).
 */
OSKAR_INLINE
void oskar_kahan_sum_complex_d(double2* sum, const double2 val, double2* c)
{
    oskar_kahan_sum_d(&sum->x, val.x, &c->x);
    oskar_kahan_sum_d(&sum->y, val.y, &c->y);
}

/**
 * @brief
 * Performs Kahan multiply-add of a complex scalar (double precision).
 *
 * @details
 * Performs Kahan summation to avoid loss of precision.
 *
 * @param[in,out] sum Updated sum.
 * @param[in]     val Value to add to sum.
 * @param[in]     f   Factor by which to multiply input value before summation.
 * @param[in,out] c   Guard value (initially 0.0; must preserve between calls).
 */
OSKAR_INLINE
void oskar_kahan_sum_multiply_complex_d(double2* sum, const double2 val,
        const double f, double2* c)
{
    oskar_kahan_sum_d(&sum->x, val.x * f, &c->x);
    oskar_kahan_sum_d(&sum->y, val.y * f, &c->y);
}

/**
 * @brief
 * Performs Kahan summation of a complex matrix (double precision).
 *
 * @details
 * Performs Kahan summation to avoid loss of precision.
 *
 * @param[in,out] sum Updated sum.
 * @param[in]     val Value to add to sum.
 * @param[in,out] c   Guard value (initially 0.0; must preserve between calls).
 */
OSKAR_INLINE
void oskar_kahan_sum_complex_matrix_d(double4c* sum, const double4c val,
        double4c* c)
{
    oskar_kahan_sum_complex_d(&sum->a, val.a, &c->a);
    oskar_kahan_sum_complex_d(&sum->b, val.b, &c->b);
    oskar_kahan_sum_complex_d(&sum->c, val.c, &c->c);
    oskar_kahan_sum_complex_d(&sum->d, val.d, &c->d);
}

/**
 * @brief
 * Performs Kahan multiply-add of a complex matrix (double precision).
 *
 * @details
 * Performs Kahan summation to avoid loss of precision.
 *
 * @param[in,out] sum Updated sum.
 * @param[in]     val Value to add to sum.
 * @param[in]     f   Factor by which to multiply input value before summation.
 * @param[in,out] c   Guard value (initially 0.0; must preserve between calls).
 */
OSKAR_INLINE
void oskar_kahan_sum_multiply_complex_matrix_d(double4c* sum,
        const double4c val, const double f, double4c* c)
{
    oskar_kahan_sum_multiply_complex_d(&sum->a, val.a, f, &c->a);
    oskar_kahan_sum_multiply_complex_d(&sum->b, val.b, f, &c->b);
    oskar_kahan_sum_multiply_complex_d(&sum->c, val.c, f, &c->c);
    oskar_kahan_sum_multiply_complex_d(&sum->d, val.d, f, &c->d);
}

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_KAHAN_SUM_H_ */
