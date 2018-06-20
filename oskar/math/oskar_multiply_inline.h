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
#include <utility/oskar_vector_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/* OUT_S = A * B */
#define OSKAR_MUL_COMPLEX(OUT_S, A, B) {                                  \
        OUT_S.x = A.x * B.x - A.y * B.y;                                  \
        OUT_S.y = A.x * B.y + A.y * B.x; }

/* OUT_S = A * conj(B) */
#define OSKAR_MUL_COMPLEX_CONJUGATE(OUT_S, A, B) {                        \
        OUT_S.x = A.x * B.x + A.y * B.y;                                  \
        OUT_S.y = A.y * B.x - A.x * B.y; }

/* A *= B */
#define OSKAR_MUL_COMPLEX_IN_PLACE(REAL2, A, B) {                         \
        const REAL2 a1__ = A;                                             \
        A.x *= B.x;         A.x -= a1__.y * B.y;                          \
        A.y = a1__.x * B.y; A.y += a1__.y * B.x; }

/* A *= conj(B) */
#define OSKAR_MUL_COMPLEX_CONJUGATE_IN_PLACE(REAL2, A, B) {               \
        const REAL2 a1__ = A;                                             \
        A.x *= B.x;         A.x += a1__.y * B.y;                          \
        A.y = a1__.y * B.x; A.y -= a1__.x * B.y; }

/* M *= A */
#define OSKAR_MUL_COMPLEX_MATRIX_COMPLEX_SCALAR_IN_PLACE(REAL2, M, A) {   \
        OSKAR_MUL_COMPLEX_IN_PLACE(REAL2, M.a, A);                        \
        OSKAR_MUL_COMPLEX_IN_PLACE(REAL2, M.b, A);                        \
        OSKAR_MUL_COMPLEX_IN_PLACE(REAL2, M.c, A);                        \
        OSKAR_MUL_COMPLEX_IN_PLACE(REAL2, M.d, A); }

/* M1 = M1 * M2
 * a = a1 a2 + b1 c2
 * b = a1 b2 + b1 d2
 * c = c1 a2 + d1 c2
 * d = c1 b2 + d1 d2 */
#define OSKAR_MUL_COMPLEX_MATRIX_IN_PLACE(REAL2, M1, M2) {                \
        REAL2 t__; const REAL2 a__ = M1.a; const REAL2 c__ = M1.c;        \
        OSKAR_MUL_COMPLEX_IN_PLACE(REAL2, M1.a, M2.a);                    \
        OSKAR_MUL_COMPLEX(t__, M1.b, M2.c);                               \
        M1.a.x += t__.x; M1.a.y += t__.y;                                 \
        OSKAR_MUL_COMPLEX_IN_PLACE(REAL2, M1.c, M2.a);                    \
        OSKAR_MUL_COMPLEX(t__, M1.d, M2.c);                               \
        M1.c.x += t__.x; M1.c.y += t__.y;                                 \
        OSKAR_MUL_COMPLEX_IN_PLACE(REAL2, M1.b, M2.d);                    \
        OSKAR_MUL_COMPLEX(t__, a__, M2.b);                                \
        M1.b.x += t__.x; M1.b.y += t__.y;                                 \
        OSKAR_MUL_COMPLEX_IN_PLACE(REAL2, M1.d, M2.d);                    \
        OSKAR_MUL_COMPLEX(t__, c__, M2.b);                                \
        M1.d.x += t__.x; M1.d.y += t__.y; }

/* M1 = M1 * conj_trans(M2) */
#define OSKAR_MUL_COMPLEX_MATRIX_CONJUGATE_TRANSPOSE_IN_PLACE(REAL2, M1, M2) { \
        REAL2 t__; const REAL2 a__ = M1.a; const REAL2 c__ = M1.c;        \
        OSKAR_MUL_COMPLEX_CONJUGATE_IN_PLACE(REAL2, M1.a, M2.a);          \
        OSKAR_MUL_COMPLEX_CONJUGATE(t__, M1.b, M2.b);                     \
        M1.a.x += t__.x; M1.a.y += t__.y;                                 \
        OSKAR_MUL_COMPLEX_CONJUGATE_IN_PLACE(REAL2, M1.c, M2.a);          \
        OSKAR_MUL_COMPLEX_CONJUGATE(t__, M1.d, M2.b);                     \
        M1.c.x += t__.x; M1.c.y += t__.y;                                 \
        OSKAR_MUL_COMPLEX_CONJUGATE_IN_PLACE(REAL2, M1.b, M2.d);          \
        OSKAR_MUL_COMPLEX_CONJUGATE(t__, a__, M2.c);                      \
        M1.b.x += t__.x; M1.b.y += t__.y;                                 \
        OSKAR_MUL_COMPLEX_CONJUGATE_IN_PLACE(REAL2, M1.d, M2.d);          \
        OSKAR_MUL_COMPLEX_CONJUGATE(t__, c__, M2.c);                      \
        M1.d.x += t__.x; M1.d.y += t__.y; }

/* M1 = M1 * M2
 * The second matrix must have a and d both real, with the form:
 *   ( a   b )
 *   ( -   d )
 */
#define OSKAR_MUL_COMPLEX_MATRIX_HERMITIAN_IN_PLACE(REAL2, M1, M2) {      \
        REAL2 t__; const REAL2 a__ = M1.a; const REAL2 c__ = M1.c;        \
        M1.a.x *= M2.a.x; M1.a.y *= M2.a.x;                               \
        OSKAR_MUL_COMPLEX_CONJUGATE(t__, M1.b, M2.b);                     \
        M1.a.x += t__.x;  M1.a.y += t__.y;                                \
        M1.c.x *= M2.a.x; M1.c.y *= M2.a.x;                               \
        OSKAR_MUL_COMPLEX_CONJUGATE(t__, M1.d, M2.b);                     \
        M1.c.x += t__.x;  M1.c.y += t__.y;                                \
        M1.b.x *= M2.d.x; M1.b.y *= M2.d.x;                               \
        OSKAR_MUL_COMPLEX(t__, a__, M2.b);                                \
        M1.b.x += t__.x;  M1.b.y += t__.y;                                \
        M1.d.x *= M2.d.x; M1.d.y *= M2.d.x;                               \
        OSKAR_MUL_COMPLEX(t__, c__, M2.b);                                \
        M1.d.x += t__.x;  M1.d.y += t__.y; }

/* OUT_S += A * B */
#define OSKAR_MUL_ADD_COMPLEX(OUT_S, A, B) {                              \
        OUT_S.x += A.x * B.x; OUT_S.x -= A.y * B.y;                       \
        OUT_S.y += A.x * B.y; OUT_S.y += A.y * B.x; }

/* OUT_S += A * conj(B) */
#define OSKAR_MUL_ADD_COMPLEX_CONJUGATE(OUT_S, A, B) {                    \
        OUT_S.x += A.x * B.x; OUT_S.x += A.y * B.y;                       \
        OUT_S.y += A.y * B.x; OUT_S.y -= A.x * B.y; }

/* OUT_M = M * A */
#define OSKAR_MUL_COMPLEX_MATRIX_COMPLEX_SCALAR(OUT_M, M, A) {            \
        OSKAR_MUL_COMPLEX(OUT_M.a, M.a, A)                                \
        OSKAR_MUL_COMPLEX(OUT_M.b, M.b, A)                                \
        OSKAR_MUL_COMPLEX(OUT_M.c, M.c, A)                                \
        OSKAR_MUL_COMPLEX(OUT_M.d, M.d, A) }

/* OUT_M += M1 * A */
#define OSKAR_MUL_ADD_COMPLEX_MATRIX_SCALAR(OUT_M, M1, A) {               \
        OUT_M.a.x += M1.a.x * A; OUT_M.a.y += M1.a.y * A;                 \
        OUT_M.b.x += M1.b.x * A; OUT_M.b.y += M1.b.y * A;                 \
        OUT_M.c.x += M1.c.x * A; OUT_M.c.y += M1.c.y * A;                 \
        OUT_M.d.x += M1.d.x * A; OUT_M.d.y += M1.d.y * A; }

/* OUT_M = M1 * M2
 * a = a1 a2 + b1 c2
 * b = a1 b2 + b1 d2
 * c = c1 a2 + d1 c2
 * d = c1 b2 + d1 d2 */
#define OSKAR_MUL_COMPLEX_MATRIX(OUT_M, M1, M2) {                         \
        OSKAR_MUL_COMPLEX(OUT_M.a, M1.a, M2.a)                            \
        OSKAR_MUL_COMPLEX(OUT_M.b, M1.a, M2.b)                            \
        OSKAR_MUL_COMPLEX(OUT_M.c, M1.c, M2.a)                            \
        OSKAR_MUL_COMPLEX(OUT_M.d, M1.c, M2.b)                            \
        OSKAR_MUL_ADD_COMPLEX(OUT_M.a, M1.b, M2.c)                        \
        OSKAR_MUL_ADD_COMPLEX(OUT_M.b, M1.b, M2.d)                        \
        OSKAR_MUL_ADD_COMPLEX(OUT_M.c, M1.d, M2.c)                        \
        OSKAR_MUL_ADD_COMPLEX(OUT_M.d, M1.d, M2.d)

/* OUT_M = M1 * conj_trans(M2)
 * a = a1 a2* + b1 b2*
 * b = a1 c2* + b1 d2*
 * c = c1 a2* + d1 b2*
 * d = c1 c2* + d1 d2* */
#define OSKAR_MUL_COMPLEX_MATRIX_CONJUGATE_TRANSPOSE(OUT_M, M1, M2) {     \
        OSKAR_MUL_COMPLEX_CONJUGATE(OUT_M.a, M1.a, M2.a)                  \
        OSKAR_MUL_COMPLEX_CONJUGATE(OUT_M.b, M1.a, M2.c)                  \
        OSKAR_MUL_COMPLEX_CONJUGATE(OUT_M.c, M1.c, M2.a)                  \
        OSKAR_MUL_COMPLEX_CONJUGATE(OUT_M.d, M1.c, M2.c)                  \
        OSKAR_MUL_ADD_COMPLEX_CONJUGATE(OUT_M.a, M1.b, M2.b)              \
        OSKAR_MUL_ADD_COMPLEX_CONJUGATE(OUT_M.b, M1.b, M2.d)              \
        OSKAR_MUL_ADD_COMPLEX_CONJUGATE(OUT_M.c, M1.d, M2.b)              \
        OSKAR_MUL_ADD_COMPLEX_CONJUGATE(OUT_M.d, M1.d, M2.d) }

/* OUT_V = M * IN */
#define OSKAR_MUL_3X3_MATRIX_VECTOR(OUT_V, M, V) {                        \
        OUT_V[0] = M[0] * V[0] + M[1] * V[1] + M[2] * V[2];               \
        OUT_V[1] = M[3] * V[0] + M[4] * V[1] + M[5] * V[2];               \
        OUT_V[2] = M[6] * V[0] + M[7] * V[1] + M[8] * V[2]; }

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

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_MULTIPLY_INLINE_H_ */
