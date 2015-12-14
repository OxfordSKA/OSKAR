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

#ifndef OSKAR_MULTIPLY_INLINE_H_
#define OSKAR_MULTIPLY_INLINE_H_

/**
 * @file oskar_multiply_inline.h
 */

#include <oskar_global.h>
#ifdef __CUDACC__
/* Must include this first to avoid type conflicts. */
#include <vector_types.h>
#endif
#include <oskar_vector_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/* OUT += P * CONJ(Q) */
#define OSKAR_MULTIPLY_ADD_COMPLEX_CONJUGATE(OUT, P, Q) \
    OUT.x += P.x * Q.x; \
    OUT.x += P.y * Q.y; \
    OUT.y += P.y * Q.x; \
    OUT.y += -(P.x * Q.y);

/**
 * @brief
 * Multiplies two complex numbers (single precision).
 *
 * @details
 * This function multiplies two complex numbers.
 *
 * @param[out] out The output complex number.
 * @param[in] a The first complex number.
 * @param[in] b The second complex number.
 */
OSKAR_INLINE
void oskar_multiply_complex_f(float2* out, const float2* a, const float2* b)
{
    out->x = a->x * b->x;
    out->y = a->x * b->y;
    out->x -= a->y * b->y; /* RE*RE - IM*IM */
    out->y += a->y * b->x; /* RE*IM + IM*RE */
}

/**
 * @brief
 * Multiplies two complex numbers (double precision).
 *
 * @details
 * This function multiplies two complex numbers.
 *
 * @param[out] out The output complex number.
 * @param[in] a The first complex number.
 * @param[in] b The second complex number.
 */
OSKAR_INLINE
void oskar_multiply_complex_d(double2* out, const double2* a, const double2* b)
{
    out->x = a->x * b->x;
    out->y = a->x * b->y;
    out->x -= a->y * b->y; /* RE*RE - IM*IM */
    out->y += a->y * b->x; /* RE*IM + IM*RE */
}

/**
 * @brief
 * Multiplies two complex numbers (single precision).
 *
 * @details
 * This function multiplies two complex numbers, taking the complex conjugate
 * of the second.
 *
 * @param[out] out The output complex number.
 * @param[in] a The first complex number.
 * @param[in] b The second complex number.
 */
OSKAR_INLINE
void oskar_multiply_complex_conjugate_f(float2* out, const float2* a,
        const float2* b)
{
    /* Multiply complex numbers a and conjugate(b). */
    out->x = a->x * b->x;
    out->y = a->y * b->x;
    out->x += a->y * b->y; /* RE*RE + IM*IM */
    out->y -= a->x * b->y; /* IM*RE - RE*IM */
}

/**
 * @brief
 * Multiplies two complex numbers (double precision).
 *
 * @details
 * This function multiplies two complex numbers, taking the complex conjugate
 * of the second.
 *
 * @param[out] out The output complex number.
 * @param[in] a The first complex number.
 * @param[in] b The second complex number.
 */
OSKAR_INLINE
void oskar_multiply_complex_conjugate_d(double2* out, const double2* a,
        const double2* b)
{
    /* Multiply complex numbers a and conjugate(b). */
    out->x = a->x * b->x;
    out->y = a->y * b->x;
    out->x += a->y * b->y; /* RE*RE + IM*IM */
    out->y -= a->x * b->y; /* IM*RE - RE*IM */
}

/**
 * @brief
 * Multiplies two complex numbers in-place (single precision).
 *
 * @details
 * This function multiplies two complex numbers, overwriting the first.
 *
 * @param[in,out] a On input, the first complex number; on output, the result.
 * @param[in] b The second complex number.
 */
OSKAR_INLINE
void oskar_multiply_complex_in_place_f(float2* a, const float2* b)
{
    /* Copy input a. */
    float2 a1;
    a1 = *a;

    /* Multiply complex numbers: a = a * b. */
    a->x *= b->x;
    a->y = a1.x * b->y;
    a->x -= a1.y * b->y; /* RE*RE - IM*IM */
    a->y += a1.y * b->x; /* RE*IM + IM*RE */
}

/**
 * @brief
 * Multiplies two complex numbers in-place (double precision).
 *
 * @details
 * This function multiplies two complex numbers, overwriting the first.
 *
 * @param[in,out] a On input, the first complex number; on output, the result.
 * @param[in] b The second complex number.
 */
OSKAR_INLINE
void oskar_multiply_complex_in_place_d(double2* a, const double2* b)
{
    /* Copy input a. */
    double2 a1;
    a1 = *a;

    /* Multiply complex numbers: a = a * b. */
    a->x *= b->x;
    a->y = a1.x * b->y;
    a->x -= a1.y * b->y; /* RE*RE - IM*IM */
    a->y += a1.y * b->x; /* RE*IM + IM*RE */
}

/**
 * @brief
 * Multiplies two complex numbers in-place (single precision).
 *
 * @details
 * This function multiplies two complex numbers, taking the complex conjugate
 * of the second, and overwriting the first.
 *
 * @param[in,out] a On input, the first complex number; on output, the result.
 * @param[in] b The second complex number.
 */
OSKAR_INLINE
void oskar_multiply_complex_conjugate_in_place_f(float2* a, const float2* b)
{
    /* Copy input a. */
    float2 a1;
    a1 = *a;

    /* Multiply complex numbers: a = a * conjugate(b). */
    a->x *= b->x;
    a->y = a1.y * b->x;
    a->x += a1.y * b->y; /* RE*RE + IM*IM */
    a->y -= a1.x * b->y; /* IM*RE - RE*IM */
}

/**
 * @brief
 * Multiplies two complex numbers in-place (double precision).
 *
 * @details
 * This function multiplies two complex numbers, taking the complex conjugate
 * of the second, and overwriting the first.
 *
 * @param[in,out] a On input, the first complex number; on output, the result.
 * @param[in] b The second complex number.
 */
OSKAR_INLINE
void oskar_multiply_complex_conjugate_in_place_d(double2* a, const double2* b)
{
    /* Copy input a. */
    double2 a1;
    a1 = *a;

    /* Multiply complex numbers: a = a * conjugate(b). */
    a->x *= b->x;
    a->y = a1.y * b->x;
    a->x += a1.y * b->y; /* RE*RE + IM*IM */
    a->y -= a1.x * b->y; /* IM*RE - RE*IM */
}

/**
 * @brief
 * Multiplies a complex matrix and a complex scalar in-place (single precision).
 *
 * @details
 * This function multiplies a complex matrix and a complex scalar, overwriting
 * the input matrix.
 *
 * @param[in,out] m On input, the complex matrix; on output, the result.
 * @param[in] a The complex scalar number.
 */
OSKAR_INLINE
void oskar_multiply_complex_matrix_complex_scalar_in_place_f(float4c* m,
        const float2* a)
{
    oskar_multiply_complex_in_place_f(&m->a, a);
    oskar_multiply_complex_in_place_f(&m->b, a);
    oskar_multiply_complex_in_place_f(&m->c, a);
    oskar_multiply_complex_in_place_f(&m->d, a);
}

/**
 * @brief
 * Multiplies a complex matrix and a complex scalar in-place (double precision).
 *
 * @details
 * This function multiplies a complex matrix and a complex scalar, overwriting
 * the input matrix.
 *
 * @param[in,out] m On input, the complex matrix; on output, the result.
 * @param[in] a The complex scalar number.
 */
OSKAR_INLINE
void oskar_multiply_complex_matrix_complex_scalar_in_place_d(double4c* m,
        const double2* a)
{
    oskar_multiply_complex_in_place_d(&m->a, a);
    oskar_multiply_complex_in_place_d(&m->b, a);
    oskar_multiply_complex_in_place_d(&m->c, a);
    oskar_multiply_complex_in_place_d(&m->d, a);
}

/**
 * @brief
 * Multiplies two complex matrices in-place (single precision).
 *
 * @details
 * This function multiplies two complex matrices, overwriting the first.
 * Matrix multiplication is done in the order M1 = M1 * M2.
 *
 * @param[in,out] m1 On input, the first complex matrix; on output, the result.
 * @param[in]     m2 The second complex matrix.
 */
OSKAR_INLINE
void oskar_multiply_complex_matrix_in_place_f(float4c* m1, const float4c* m2)
{
    /* Copy a and c from the input matrix. */
    float2 a, c, t;
    a = m1->a;
    c = m1->c;

    oskar_multiply_complex_in_place_f(&m1->a, &m2->a); /* a = a1 a2 + b1 c2 */
    oskar_multiply_complex_f(&t, &m1->b, &m2->c);
    m1->a.x += t.x; /* Real part. */
    m1->a.y += t.y; /* Imag part. */
    oskar_multiply_complex_in_place_f(&m1->c, &m2->a); /* c = c1 a2 + d1 c2 */
    oskar_multiply_complex_f(&t, &m1->d, &m2->c);
    m1->c.x += t.x; /* Real part. */
    m1->c.y += t.y; /* Imag part. */
    oskar_multiply_complex_in_place_f(&m1->b, &m2->d); /* b = a1 b2 + b1 d2 */
    oskar_multiply_complex_f(&t, &a, &m2->b);
    m1->b.x += t.x; /* Real part. */
    m1->b.y += t.y; /* Imag part. */
    oskar_multiply_complex_in_place_f(&m1->d, &m2->d); /* d = c1 b2 + d1 d2 */
    oskar_multiply_complex_f(&t, &c, &m2->b);
    m1->d.x += t.x; /* Real part. */
    m1->d.y += t.y; /* Imag part. */
}

/**
 * @brief
 * Multiplies two complex matrices in-place (double precision).
 *
 * @details
 * This function multiplies two complex matrices, overwriting the first.
 * Matrix multiplication is done in the order M1 = M1 * M2.
 *
 * @param[in,out] m1 On input, the first complex matrix; on output, the result.
 * @param[in]     m2 The second complex matrix.
 */
OSKAR_INLINE
void oskar_multiply_complex_matrix_in_place_d(double4c* m1, const double4c* m2)
{
    /* Copy a and c from the input matrix. */
    double2 a, c, t;
    a = m1->a;
    c = m1->c;

    oskar_multiply_complex_in_place_d(&m1->a, &m2->a); /* a = a1 a2 + b1 c2 */
    oskar_multiply_complex_d(&t, &m1->b, &m2->c);
    m1->a.x += t.x; /* Real part. */
    m1->a.y += t.y; /* Imag part. */
    oskar_multiply_complex_in_place_d(&m1->c, &m2->a); /* c = c1 a2 + d1 c2 */
    oskar_multiply_complex_d(&t, &m1->d, &m2->c);
    m1->c.x += t.x; /* Real part. */
    m1->c.y += t.y; /* Imag part. */
    oskar_multiply_complex_in_place_d(&m1->b, &m2->d); /* b = a1 b2 + b1 d2 */
    oskar_multiply_complex_d(&t, &a, &m2->b);
    m1->b.x += t.x; /* Real part. */
    m1->b.y += t.y; /* Imag part. */
    oskar_multiply_complex_in_place_d(&m1->d, &m2->d); /* d = c1 b2 + d1 d2 */
    oskar_multiply_complex_d(&t, &c, &m2->b);
    m1->d.x += t.x; /* Real part. */
    m1->d.y += t.y; /* Imag part. */
}

/**
 * @brief
 * Multiplies two complex matrices, first taking the conjugate transpose of
 * the second (single precision).
 *
 * @details
 * This function multiplies together two complex matrices.
 * The Hermitian conjugate of the second matrix is taken before the
 * multiplication. Matrix multiplication is done in the order M1 = M1 * M2^H.
 *
 * @param[in] m1 On input, the first complex matrix; on output, the result.
 * @param[in] m2 The second complex matrix.
 */
OSKAR_INLINE
void oskar_multiply_complex_matrix_conjugate_transpose_in_place_f(float4c* m1,
        const float4c* m2)
{
    /* Copy a and c from the input matrix. */
    float2 a, c, t;
    a = m1->a;
    c = m1->c;

    /* First, evaluate result a. */
    oskar_multiply_complex_conjugate_in_place_f(&m1->a, &m2->a);
    oskar_multiply_complex_conjugate_f(&t, &m1->b, &m2->b);
    m1->a.x += t.x; /* Real part. */
    m1->a.y += t.y; /* Imag part. */

    /* Second, evaluate result c. */
    oskar_multiply_complex_conjugate_in_place_f(&m1->c, &m2->a);
    oskar_multiply_complex_conjugate_f(&t, &m1->d, &m2->b);
    m1->c.x += t.x; /* Real part. */
    m1->c.y += t.y; /* Imag part. */

    /* Third, evaluate result b. */
    oskar_multiply_complex_conjugate_in_place_f(&m1->b, &m2->d);
    oskar_multiply_complex_conjugate_f(&t, &a, &m2->c);
    m1->b.x += t.x; /* Real part. */
    m1->b.y += t.y; /* Imag part. */

    /* Fourth, evaluate result d. */
    oskar_multiply_complex_conjugate_in_place_f(&m1->d, &m2->d);
    oskar_multiply_complex_conjugate_f(&t, &c, &m2->c);
    m1->d.x += t.x; /* Real part. */
    m1->d.y += t.y; /* Imag part. */
}

/**
 * @brief
 * Multiplies two complex matrices, first taking the conjugate transpose of
 * the second (double precision).
 *
 * @details
 * This function multiplies together two complex matrices.
 * The Hermitian conjugate of the second matrix is taken before the
 * multiplication. Matrix multiplication is done in the order M1 = M1 * M2^H.
 *
 * @param[in] m1 On input, the first complex matrix; on output, the result.
 * @param[in] m2 The second complex matrix.
 */
OSKAR_INLINE
void oskar_multiply_complex_matrix_conjugate_transpose_in_place_d(double4c* m1,
        const double4c* m2)
{
    /* Copy a and c from the input matrix. */
    double2 a, c, t;
    a = m1->a;
    c = m1->c;

    /* First, evaluate result a. */
    oskar_multiply_complex_conjugate_in_place_d(&m1->a, &m2->a);
    oskar_multiply_complex_conjugate_d(&t, &m1->b, &m2->b);
    m1->a.x += t.x; /* Real part. */
    m1->a.y += t.y; /* Imag part. */

    /* Second, evaluate result c. */
    oskar_multiply_complex_conjugate_in_place_d(&m1->c, &m2->a);
    oskar_multiply_complex_conjugate_d(&t, &m1->d, &m2->b);
    m1->c.x += t.x; /* Real part. */
    m1->c.y += t.y; /* Imag part. */

    /* Third, evaluate result b. */
    oskar_multiply_complex_conjugate_in_place_d(&m1->b, &m2->d);
    oskar_multiply_complex_conjugate_d(&t, &a, &m2->c);
    m1->b.x += t.x; /* Real part. */
    m1->b.y += t.y; /* Imag part. */

    /* Fourth, evaluate result d. */
    oskar_multiply_complex_conjugate_in_place_d(&m1->d, &m2->d);
    oskar_multiply_complex_conjugate_d(&t, &c, &m2->c);
    m1->d.x += t.x; /* Real part. */
    m1->d.y += t.y; /* Imag part. */
}

/**
 * @brief
 * Multiplies a complex matrix and a Hermitian matrix (single precision).
 *
 * @details
 * This function multiplies together two complex 2x2 matrices,
 * where the second one is Hermitian.
 *
 * The second matrix is assumed to contain values as follows:
 *
 *   ( a   b )
 *   ( -   d )
 *
 * and a and d must both be real.
 *
 * Matrix multiplication is done in the order M1 = M1 * M2.
 *
 * @param[in,out] m1 On input, the complex matrix; on output, the result.
 * @param[in]     m2 The Hermitian matrix.
 */
OSKAR_INLINE
void oskar_multiply_complex_matrix_hermitian_in_place_f(float4c* m1,
        const float4c* m2)
{
    /* Copy a and c from the input matrix. */
    float2 a, c, t;
    a = m1->a;
    c = m1->c;

    /* First, evaluate result a. */
    m1->a.x *= m2->a.x;
    m1->a.y *= m2->a.x;
    oskar_multiply_complex_conjugate_f(&t, &m1->b, &m2->b);
    m1->a.x += t.x; /* Real part. */
    m1->a.y += t.y; /* Imag part. */

    /* Second, evaluate result c. */
    m1->c.x *= m2->a.x;
    m1->c.y *= m2->a.x;
    oskar_multiply_complex_conjugate_f(&t, &m1->d, &m2->b);
    m1->c.x += t.x; /* Real part. */
    m1->c.y += t.y; /* Imag part. */

    /* Third, evaluate result b. */
    m1->b.x *= m2->d.x;
    m1->b.y *= m2->d.x;
    oskar_multiply_complex_f(&t, &a, &m2->b);
    m1->b.x += t.x; /* Real part. */
    m1->b.y += t.y; /* Imag part. */

    /* Fourth, evaluate result d. */
    m1->d.x *= m2->d.x;
    m1->d.y *= m2->d.x;
    oskar_multiply_complex_f(&t, &c, &m2->b);
    m1->d.x += t.x; /* Real part. */
    m1->d.y += t.y; /* Imag part. */
}

/**
 * @brief
 * Multiplies a complex matrix and a Hermitian matrix (double precision).
 *
 * @details
 * This function multiplies together two complex 2x2 matrices,
 * where the second one is Hermitian.
 *
 * The second matrix is assumed to contain values as follows:
 *
 *   ( a   b )
 *   ( -   d )
 *
 * and a and d must both be real.
 *
 * Matrix multiplication is done in the order M1 = M1 * M2.
 *
 * @param[in,out] m1 On input, the complex matrix; on output, the result.
 * @param[in]     m2 The Hermitian matrix.
 */
OSKAR_INLINE
void oskar_multiply_complex_matrix_hermitian_in_place_d(double4c* m1,
        const double4c* m2)
{
    /* Copy a and c from the input matrix. */
    double2 a, c, t;
    a = m1->a;
    c = m1->c;

    /* First, evaluate result a. */
    m1->a.x *= m2->a.x;
    m1->a.y *= m2->a.x;
    oskar_multiply_complex_conjugate_d(&t, &m1->b, &m2->b);
    m1->a.x += t.x; /* Real part. */
    m1->a.y += t.y; /* Imag part. */

    /* Second, evaluate result c. */
    m1->c.x *= m2->a.x;
    m1->c.y *= m2->a.x;
    oskar_multiply_complex_conjugate_d(&t, &m1->d, &m2->b);
    m1->c.x += t.x; /* Real part. */
    m1->c.y += t.y; /* Imag part. */

    /* Third, evaluate result b. */
    m1->b.x *= m2->d.x;
    m1->b.y *= m2->d.x;
    oskar_multiply_complex_d(&t, &a, &m2->b);
    m1->b.x += t.x; /* Real part. */
    m1->b.y += t.y; /* Imag part. */

    /* Fourth, evaluate result d. */
    m1->d.x *= m2->d.x;
    m1->d.y *= m2->d.x;
    oskar_multiply_complex_d(&t, &c, &m2->b);
    m1->d.x += t.x; /* Real part. */
    m1->d.y += t.y; /* Imag part. */
}

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_MULTIPLY_INLINE_H_ */
