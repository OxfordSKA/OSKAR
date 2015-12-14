/*
 * Copyright (c) 2013-2015, The University of Oxford
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

#ifndef OSKAR_PRIVATE_CORRELATE_FUNCTIONS_INLINE_H_
#define OSKAR_PRIVATE_CORRELATE_FUNCTIONS_INLINE_H_

#include <oskar_cmath.h>
#include <oskar_global.h>
#include <oskar_multiply_inline.h>
#include <oskar_kahan_sum.h>

#define OMEGA_EARTH  7.272205217e-5  /* radians/sec */
#define OMEGA_EARTHf 7.272205217e-5f /* radians/sec */

#ifdef __cplusplus
extern "C" {
#endif

/* Construct source brightness matrix (ignoring c, as it's Hermitian). */
#define OSKAR_CONSTRUCT_B_FLOAT(B, I, Q, U, V, S) { \
        float s_I, s_Q; s_I = I[S]; s_Q = Q[S]; \
        B.b.x = U[S]; \
        B.b.y = V[S]; \
        B.a.x = s_I + s_Q; \
        B.d.x = s_I - s_Q; }

/* Construct source brightness matrix (ignoring c, as it's Hermitian). */
#define OSKAR_CONSTRUCT_B_DOUBLE(B, I, Q, U, V, S) { \
        double s_I, s_Q; s_I = I[S]; s_Q = Q[S]; \
        B.b.x = U[S]; \
        B.b.y = V[S]; \
        B.a.x = s_I + s_Q; \
        B.d.x = s_I - s_Q; }

#define OSKAR_ADD_TO_VIS_POL_SMEAR(V, M, F) \
        V->a.x += M.a.x * F; \
        V->a.y += M.a.y * F; \
        V->b.x += M.b.x * F; \
        V->b.y += M.b.y * F; \
        V->c.x += M.c.x * F; \
        V->c.y += M.c.y * F; \
        V->d.x += M.d.x * F; \
        V->d.y += M.d.y * F;

#define OSKAR_ADD_TO_VIS_POL(V, M) \
        V->a.x += M.a.x; \
        V->a.y += M.a.y; \
        V->b.x += M.b.x; \
        V->b.y += M.b.y; \
        V->c.x += M.c.x; \
        V->c.y += M.c.y; \
        V->d.x += M.d.x; \
        V->d.y += M.d.y;

#if defined(__CUDACC__) && __CUDA_ARCH__ >= 350
/* Uses __ldg() instruction for global load via texture cache. */
/* Available only from CUDA architecture 3.5. */
#define OSKAR_LOAD_MATRIX(M, J, IDX) \
        M.a = __ldg(&(J[IDX].a)); \
        M.b = __ldg(&(J[IDX].b)); \
        M.c = __ldg(&(J[IDX].c)); \
        M.d = __ldg(&(J[IDX].d));
#else
#define OSKAR_LOAD_MATRIX(M, J, IDX) M = J[IDX];
#endif

/**
 * @brief
 * Function to evaluate sinc(x) (single precision).
 *
 * @details
 * This function evaluates sinc(x) = sin(x) / x.
 *
 * @param[in] x Function argument.
 */
OSKAR_INLINE
float oskar_sinc_f(const float a)
{
    return (a == 0.0f) ? 1.0f : sinf(a) / a;
}

/**
 * @brief
 * Function to evaluate sinc(x) (double precision).
 *
 * @details
 * This function evaluates sinc(x) = sin(x) / x.
 *
 * @param[in] x Function argument.
 */
OSKAR_INLINE
double oskar_sinc_d(const double a)
{
    return (a == 0.0) ? 1.0 : sin(a) / a;
}

/**
 * @brief
 * Accumulates the visibility response on one baseline due to a single source
 * (single precision).
 *
 * @details
 * This function evaluates the visibility response for a single source on a
 * single baseline between stations p and q, accumulating the result with
 * the existing baseline visibility. It requires the final (collapsed)
 * Jones matrices for the source for stations p and q, and the unmodified
 * Stokes parameters of the source.
 *
 * The output visibility is updated according to:
 *
 * V_pq = V_pq + J_p * B * J_q^H
 *
 * where the brightness matrix B is assembled as:
 *
 * B = [ I + Q    U + iV ]
 *     [ U - iV    I - Q ]
 *
 * Visibilities are updated for a single source, even though array pointers are
 * passed to this function. The source is specified using the \p source_id
 * index parameter.
 *
 * @param[in,out] V_pq  Running total of source visibilities.
 * @param[in] source_id Index of source in all arrays.
 * @param[in] I         Array of source Stokes I values, in Jy.
 * @param[in] Q         Array of source Stokes Q values, in Jy.
 * @param[in] U         Array of source Stokes U values, in Jy.
 * @param[in] V         Array of source Stokes V values, in Jy.
 * @param[in] J_p       Array of source Jones matrices for station p.
 * @param[in] J_q       Array of source Jones matrices for station q.
 * @param[in] smear     Smearing factor by which to modify source visibility.
 * @param[in,out] guard Updated guard value used in Kahan summation.
 */
OSKAR_INLINE
void oskar_accumulate_baseline_visibility_for_source_inline_f(
        float4c* restrict V_pq, const int source_id,
        const float* restrict I, const float* restrict Q,
        const float* restrict U, const float* restrict V,
        const float4c* restrict J_p, const float4c* restrict J_q,
        const float smear
#ifndef __CUDACC__
        , float4c* restrict guard
#endif
        )
{
    float4c m1, m2;

    /* Construct source brightness matrix. */
    OSKAR_CONSTRUCT_B_FLOAT(m2, I, Q, U, V, source_id)

    /* Multiply first Jones matrix with source brightness matrix. */
    OSKAR_LOAD_MATRIX(m1, J_p, source_id)
    oskar_multiply_complex_matrix_hermitian_in_place_f(&m1, &m2);

    /* Multiply result with second (Hermitian transposed) Jones matrix. */
    OSKAR_LOAD_MATRIX(m2, J_q, source_id)
    oskar_multiply_complex_matrix_conjugate_transpose_in_place_f(&m1, &m2);

#ifdef __CUDACC__
    /* Multiply result by smearing term and accumulate. */
    OSKAR_ADD_TO_VIS_POL_SMEAR(V_pq, m1, smear)
#else
    oskar_kahan_sum_multiply_complex_matrix_f(V_pq, m1, smear, guard);
#endif /* __CUDACC__ */
}

/**
 * @brief
 * Accumulates the visibility response on one baseline due to a single source
 * (single precision).
 *
 * @details
 * This function evaluates the visibility response for a single source on a
 * single baseline between stations p and q, accumulating the result with
 * the existing baseline visibility. It requires the final (collapsed)
 * Jones matrices for the source for stations p and q.
 *
 * The output visibility is updated according to:
 *
 * V_pq = V_pq + J_p * J_q^H
 *
 * Visibilities are updated for a single source, even though array pointers are
 * passed to this function. The source is specified using the \p source_id
 * index parameter.
 *
 * @param[in,out] V_pq  Running total of source visibilities.
 * @param[in] source_id Index of source in all arrays.
 * @param[in] J_p       Array of source Jones matrices for station p.
 * @param[in] J_q       Array of source Jones matrices for station q.
 * @param[in] smear     Smearing factor by which to modify source visibility.
 * @param[in,out] guard Updated guard value used in Kahan summation.
 */
OSKAR_INLINE
void oskar_accumulate_baseline_visibility_for_source_inline_new_f(
        float4c* restrict V_pq, const int source_id,
        const float4c* restrict J_p, const float4c* restrict J_q,
        const float smear
#ifndef __CUDACC__
        , float4c* restrict guard
#endif
        )
{
    float4c m1, m2;

    /* Get Jones matrices. */
    OSKAR_LOAD_MATRIX(m1, J_p, source_id)
    OSKAR_LOAD_MATRIX(m2, J_q, source_id)

    /* Multiply Jones matrices. */
    oskar_multiply_complex_matrix_conjugate_transpose_in_place_f(&m1, &m2);

#ifdef __CUDACC__
    /* Multiply result by smearing term and accumulate. */
    OSKAR_ADD_TO_VIS_POL_SMEAR(V_pq, m1, smear)
#else
    oskar_kahan_sum_multiply_complex_matrix_f(V_pq, m1, smear, guard);
#endif /* __CUDACC__ */
}

/**
 * @brief
 * Accumulates the visibility response on one baseline due to a single source
 * (double precision).
 *
 * @details
 * This function evaluates the visibility response for a single source on a
 * single baseline between stations p and q, accumulating the result with
 * the existing baseline visibility. It requires the final (collapsed)
 * Jones matrices for the source for stations p and q, and the unmodified
 * Stokes parameters of the source.
 *
 * The output visibility is updated according to:
 *
 * V_pq = V_pq + J_p * B * J_q^H
 *
 * where the brightness matrix B is assembled as:
 *
 * B = [ I + Q    U + iV ]
 *     [ U - iV    I - Q ]
 *
 * Visibilities are updated for a single source, even though array pointers are
 * passed to this function. The source is specified using the \p source_id
 * index parameter.
 *
 * @param[in,out] V_pq  Running total of source visibilities.
 * @param[in] source_id Index of source in all arrays.
 * @param[in] I         Array of source Stokes I values, in Jy.
 * @param[in] Q         Array of source Stokes Q values, in Jy.
 * @param[in] U         Array of source Stokes U values, in Jy.
 * @param[in] V         Array of source Stokes V values, in Jy.
 * @param[in] J_p       Array of source Jones matrices for station p.
 * @param[in] J_q       Array of source Jones matrices for station q.
 * @param[in] smear     Smearing factor by which to modify source visibility.
 */
OSKAR_INLINE
void oskar_accumulate_baseline_visibility_for_source_inline_d(
        double4c* restrict V_pq, const int source_id,
        const double* restrict I, const double* restrict Q,
        const double* restrict U, const double* restrict V,
        const double4c* restrict J_p, const double4c* restrict J_q,
        const double smear)
{
    double4c m1, m2;

    /* Construct source brightness matrix. */
    OSKAR_CONSTRUCT_B_DOUBLE(m2, I, Q, U, V, source_id)

    /* Multiply first Jones matrix with source brightness matrix. */
    OSKAR_LOAD_MATRIX(m1, J_p, source_id)
    oskar_multiply_complex_matrix_hermitian_in_place_d(&m1, &m2);

    /* Multiply result with second (Hermitian transposed) Jones matrix. */
    OSKAR_LOAD_MATRIX(m2, J_q, source_id)
    oskar_multiply_complex_matrix_conjugate_transpose_in_place_d(&m1, &m2);

    /* Multiply result by smearing term and accumulate. */
    OSKAR_ADD_TO_VIS_POL_SMEAR(V_pq, m1, smear)
}

/**
 * @brief
 * Accumulates the visibility response on one baseline due to a single source
 * (double precision).
 *
 * @details
 * This function evaluates the visibility response for a single source on a
 * single baseline between stations p and q, accumulating the result with
 * the existing baseline visibility. It requires the final (collapsed)
 * Jones matrices for the source for stations p and q.
 *
 * The output visibility is updated according to:
 *
 * V_pq = V_pq + J_p * J_q^H
 *
 * Visibilities are updated for a single source, even though array pointers are
 * passed to this function. The source is specified using the \p source_id
 * index parameter.
 *
 * @param[in,out] V_pq  Running total of source visibilities.
 * @param[in] source_id Index of source in all arrays.
 * @param[in] J_p       Array of source Jones matrices for station p.
 * @param[in] J_q       Array of source Jones matrices for station q.
 * @param[in] smear     Smearing factor by which to modify source visibility.
 */
OSKAR_INLINE
void oskar_accumulate_baseline_visibility_for_source_inline_new_d(
        double4c* restrict V_pq, const int source_id,
        const double4c* restrict J_p, const double4c* restrict J_q,
        const double smear)
{
    double4c m1, m2;

    /* Get Jones matrices. */
    OSKAR_LOAD_MATRIX(m1, J_p, source_id)
    OSKAR_LOAD_MATRIX(m2, J_q, source_id)

    /* Multiply Jones matrices. */
    oskar_multiply_complex_matrix_conjugate_transpose_in_place_d(&m1, &m2);

    /* Multiply result by smearing term and accumulate. */
    OSKAR_ADD_TO_VIS_POL_SMEAR(V_pq, m1, smear)
}

/**
 * @brief
 * Accumulates the visibility response on one baseline due to a single source
 * (scalar, single precision).
 *
 * @details
 * This function evaluates the visibility response for a single source on a
 * single baseline between stations p and q, accumulating the result with
 * the existing baseline visibility. It requires the final (collapsed)
 * Jones scalars for the source for stations p and q, and the unmodified
 * Stokes I value of the source.
 *
 * The output visibility is updated according to:
 *
 * V_pq = V_pq + J_p * I * J_q^H
 *
 * Visibilities are updated for a single source, even though array pointers are
 * passed to this function. The source is specified using the \p source_id
 * index parameter.
 *
 * @param[in,out] V_pq  Running total of source visibilities.
 * @param[in] source_id Index of source in all arrays.
 * @param[in] I         Array of source Stokes I values, in Jy.
 * @param[in] J_p       Array of source Jones matrices for station p.
 * @param[in] J_q       Array of source Jones matrices for station q.
 * @param[in] smear     Smearing factor by which to modify source visibility.
 * @param[in,out] guard Updated guard value used in Kahan summation.
 */
OSKAR_INLINE
void oskar_accumulate_baseline_visibility_for_source_scalar_inline_f(
        float2* restrict V_pq, const int source_id, const float* restrict I,
        const float2* restrict J_p, const float2* restrict J_q,
        const float smear
#ifndef __CUDACC__
        , float2* restrict guard
#endif
        )
{
    float2 t1, t2;
    float I_;

    /* Multiply first Jones scalar with Stokes I value. */
    I_ = I[source_id];
    t1 = J_p[source_id];
    t1.x *= I_;
    t1.y *= I_;

    /* Multiply result with second (conjugated) Jones scalar. */
    t2 = J_q[source_id];
    oskar_multiply_complex_conjugate_in_place_f(&t1, &t2);

    /* Multiply result by smearing term and accumulate. */
#ifdef __CUDACC__
    V_pq->x += t1.x * smear;
    V_pq->y += t1.y * smear;
#else
    oskar_kahan_sum_multiply_complex_f(V_pq, t1, smear, guard);
#endif /* __CUDACC__ */
}

/**
 * @brief
 * Accumulates the visibility response on one baseline due to a single source
 * (scalar, double precision).
 *
 * @details
 * This function evaluates the visibility response for a single source on a
 * single baseline between stations p and q, accumulating the result with
 * the existing baseline visibility. It requires the final (collapsed)
 * Jones scalars for the source for stations p and q, and the unmodified
 * Stokes I value of the source.
 *
 * The output visibility is updated according to:
 *
 * V_pq = V_pq + J_p * I * J_q^H
 *
 * Visibilities are updated for a single source, even though array pointers are
 * passed to this function. The source is specified using the \p source_id
 * index parameter.
 *
 * @param[in,out] V_pq  Running total of source visibilities.
 * @param[in] source_id Index of source in all arrays.
 * @param[in] I         Array of source Stokes I values, in Jy.
 * @param[in] J_p       Array of source Jones matrices for station p.
 * @param[in] J_q       Array of source Jones matrices for station q.
 * @param[in] smear     Smearing factor by which to modify source visibility.
 */
OSKAR_INLINE
void oskar_accumulate_baseline_visibility_for_source_scalar_inline_d(
        double2* restrict V_pq, const int source_id, const double* restrict I,
        const double2* restrict J_p, const double2* restrict J_q,
        const double smear)
{
    double2 t1, t2;
    double I_;

    /* Multiply first Jones scalar with Stokes I value. */
    I_ = I[source_id];
    t1 = J_p[source_id];
    t1.x *= I_;
    t1.y *= I_;

    /* Multiply result with second (conjugated) Jones scalar. */
    t2 = J_q[source_id];
    oskar_multiply_complex_conjugate_in_place_d(&t1, &t2);

    /* Multiply result by smearing term and accumulate. */
    V_pq->x += t1.x * smear;
    V_pq->y += t1.y * smear;
}

/**
 * @brief
 * Accumulates the visibility response at one station due to a single source
 * (single precision).
 *
 * @details
 * This function evaluates the visibility response for a single source at a
 * single station, accumulating the result with the existing station
 * visibility. It requires the final (collapsed) Jones matrices for the
 * source and station, and the unmodified Stokes parameters of the source.
 *
 * The output visibility is updated according to:
 *
 * V = V + J * B * J^H
 *
 * where the brightness matrix B is assembled as:
 *
 * B = [ I + Q    U + iV ]
 *     [ U - iV    I - Q ]
 *
 * Visibilities are updated for a single source, even though array pointers are
 * passed to this function. The source is specified using the \p source_id
 * index parameter.
 *
 * @param[in,out] vis   Running total of source visibilities.
 * @param[in] source_id Index of source in all arrays.
 * @param[in] I         Array of source Stokes I values, in Jy.
 * @param[in] Q         Array of source Stokes Q values, in Jy.
 * @param[in] U         Array of source Stokes U values, in Jy.
 * @param[in] V         Array of source Stokes V values, in Jy.
 * @param[in] J         Array of source Jones matrices for station.
 * @param[in,out] guard Updated guard value used in Kahan summation.
 */
OSKAR_INLINE
void oskar_accumulate_station_visibility_for_source_inline_f(
        float4c* restrict vis, const int source_id,
        const float* restrict I, const float* restrict Q,
        const float* restrict U, const float* restrict V,
        const float4c* restrict J
#ifndef __CUDACC__
        , float4c* restrict guard
#endif
        )
{
    float4c m1, m2;

    /* Construct source brightness matrix. */
    OSKAR_CONSTRUCT_B_FLOAT(m2, I, Q, U, V, source_id)

    /* Multiply first Jones matrix with source brightness matrix. */
    OSKAR_LOAD_MATRIX(m1, J, source_id)
    oskar_multiply_complex_matrix_hermitian_in_place_f(&m1, &m2);

    /* Multiply result with second (Hermitian transposed) Jones matrix. */
    OSKAR_LOAD_MATRIX(m2, J, source_id)
    oskar_multiply_complex_matrix_conjugate_transpose_in_place_f(&m1, &m2);

#ifdef __CUDACC__
    /* Accumulate. */
    OSKAR_ADD_TO_VIS_POL(vis, m1)
#else
    oskar_kahan_sum_complex_matrix_f(vis, m1, guard);
#endif /* __CUDACC__ */
}

/**
 * @brief
 * Accumulates the visibility response for one station due to a single source
 * (scalar, single precision).
 *
 * @details
 * This function evaluates the visibility response for a single source at one
 * station, accumulating the result with the existing visibility.
 * It requires the final (collapsed) Jones scalars for the source, and the
 * unmodified Stokes I value of the source.
 *
 * The output visibility is updated according to:
 *
 * V = V + J * I * J^H
 *
 * Visibilities are updated for a single source, even though array pointers are
 * passed to this function. The source is specified using the \p source_id
 * index parameter.
 *
 * @param[in,out] vis   Running total of source visibilities.
 * @param[in] source_id Index of source in all arrays.
 * @param[in] I         Array of source Stokes I values, in Jy.
 * @param[in] J         Array of source Jones matrices for station p.
 * @param[in,out] guard Updated guard value used in Kahan summation.
 */
OSKAR_INLINE
void oskar_accumulate_station_visibility_for_source_scalar_inline_f(
        float2* restrict vis, const int source_id, const float* restrict I,
        const float2* restrict J
#ifndef __CUDACC__
        , float2* restrict guard
#endif
        )
{
    float2 t1, t2;
    float I_;

    /* Multiply. */
    I_ = I[source_id];
    t1 = J[source_id];
    t2 = t1;
    oskar_multiply_complex_conjugate_in_place_f(&t1, &t2);
    t1.x *= I_;
    t1.y *= I_;

    /* Accumulate. */
#ifdef __CUDACC__
    vis->x += t1.x;
    vis->y += t1.y;
#else
    oskar_kahan_sum_complex_f(vis, t1, guard);
#endif /* __CUDACC__ */
}

/**
 * @brief
 * Accumulates the visibility response at one station due to a single source
 * (double precision).
 *
 * @details
 * This function evaluates the visibility response for a single source at a
 * single station, accumulating the result with the existing station
 * visibility. It requires the final (collapsed) Jones matrices for the
 * source and station, and the unmodified Stokes parameters of the source.
 *
 * The output visibility is updated according to:
 *
 * V = V + J * B * J^H
 *
 * where the brightness matrix B is assembled as:
 *
 * B = [ I + Q    U + iV ]
 *     [ U - iV    I - Q ]
 *
 * Visibilities are updated for a single source, even though array pointers are
 * passed to this function. The source is specified using the \p source_id
 * index parameter.
 *
 * @param[in,out] vis   Running total of source visibilities.
 * @param[in] source_id Index of source in all arrays.
 * @param[in] I         Array of source Stokes I values, in Jy.
 * @param[in] Q         Array of source Stokes Q values, in Jy.
 * @param[in] U         Array of source Stokes U values, in Jy.
 * @param[in] V         Array of source Stokes V values, in Jy.
 * @param[in] J         Array of source Jones matrices for station.
 */
OSKAR_INLINE
void oskar_accumulate_station_visibility_for_source_inline_d(
        double4c* restrict vis, const int source_id,
        const double* restrict I, const double* restrict Q,
        const double* restrict U, const double* restrict V,
        const double4c* restrict J)
{
    double4c m1, m2;

    /* Construct source brightness matrix. */
    OSKAR_CONSTRUCT_B_DOUBLE(m2, I, Q, U, V, source_id)

    /* Multiply first Jones matrix with source brightness matrix. */
    OSKAR_LOAD_MATRIX(m1, J, source_id)
    oskar_multiply_complex_matrix_hermitian_in_place_d(&m1, &m2);

    /* Multiply result with second (Hermitian transposed) Jones matrix. */
    OSKAR_LOAD_MATRIX(m2, J, source_id)
    oskar_multiply_complex_matrix_conjugate_transpose_in_place_d(&m1, &m2);

    /* Accumulate. */
    OSKAR_ADD_TO_VIS_POL(vis, m1)
}

/**
 * @brief
 * Accumulates the visibility response for one station due to a single source
 * (scalar, double precision).
 *
 * @details
 * This function evaluates the visibility response for a single source at one
 * station, accumulating the result with the existing visibility.
 * It requires the final (collapsed) Jones scalars for the source, and the
 * unmodified Stokes I value of the source.
 *
 * The output visibility is updated according to:
 *
 * V = V + J * I * J^H
 *
 * Visibilities are updated for a single source, even though array pointers are
 * passed to this function. The source is specified using the \p source_id
 * index parameter.
 *
 * @param[in,out] vis   Running total of source visibilities.
 * @param[in] source_id Index of source in all arrays.
 * @param[in] I         Array of source Stokes I values, in Jy.
 * @param[in] J         Array of source Jones matrices for station p.
 */
OSKAR_INLINE
void oskar_accumulate_station_visibility_for_source_scalar_inline_d(
        double2* restrict vis, const int source_id, const double* restrict I,
        const double2* restrict J)
{
    double2 t1, t2;
    double I_;

    /* Multiply. */
    I_ = I[source_id];
    t1 = J[source_id];
    t2 = t1;
    oskar_multiply_complex_conjugate_in_place_d(&t1, &t2);
    t1.x *= I_;
    t1.y *= I_;

    /* Accumulate. */
    vis->x += t1.x;
    vis->y += t1.y;
}

/**
 * @brief
 * Evaluates the rate of change of baseline coordinates with time
 * (single precision).
 *
 * @details
 * This function evaluates the rate of change of the baseline coordinates
 * with time, due to Earth rotation.
 *
 * @param[in] station_xp     Offset ECEF x-coordinate of station p, in metres.
 * @param[in] station_xq     Offset ECEF x-coordinate of station q, in metres.
 * @param[in] station_yp     Offset ECEF y-coordinate of station p, in metres.
 * @param[in] station_yq     Offset ECEF y-coordinate of station q, in metres.
 * @param[in] inv_wavelength Inverse of the wavelength, in metres.
 * @param[in] time_int_sec   Time averaging interval, in seconds.
 * @param[in] gha0_rad       Greenwich Hour Angle of phase centre, in radians.
 * @param[in] dec0_rad       Declination of phase centre, in radians.
 * @param[out] du_dt         Rate of change of baseline u coordinate, in m/s.
 * @param[out] dv_dt         Rate of change of baseline v coordinate, in m/s.
 * @param[out] dw_dt         Rate of change of baseline w coordinate, in m/s.
 */
OSKAR_INLINE
void oskar_evaluate_baseline_derivatives_inline_f(const float station_xp,
        const float station_xq, const float station_yp,
        const float station_yq, const float inv_wavelength,
        const float time_int_sec, const float gha0_rad,
        const float dec0_rad, float* du_dt, float* dv_dt, float* dw_dt)
{
    float xx, yy, rot_angle, temp, sin_HA, cos_HA, sin_Dec, cos_Dec;
#ifdef __CUDACC__
    sincosf(gha0_rad, &sin_HA, &cos_HA);
    sincosf(dec0_rad, &sin_Dec, &cos_Dec);
#else
    sin_HA = sinf(gha0_rad);
    cos_HA = cosf(gha0_rad);
    sin_Dec = sinf(dec0_rad);
    cos_Dec = cosf(dec0_rad);
#endif
    temp = M_PIf * inv_wavelength;
    xx = (station_xp - station_xq) * temp;
    yy = (station_yp - station_yq) * temp;
    rot_angle = OMEGA_EARTHf * time_int_sec;
    temp = (xx * sin_HA + yy * cos_HA) * rot_angle;
    *du_dt = (xx * cos_HA - yy * sin_HA) * rot_angle;
    *dv_dt = temp * sin_Dec;
    *dw_dt = -temp * cos_Dec;
}

/**
 * @brief
 * Evaluates the rate of change of baseline coordinates with time
 * (double precision).
 *
 * @details
 * This function evaluates the rate of change of the baseline coordinates
 * with time, due to Earth rotation.
 *
 * @param[in] station_xp     Offset ECEF x-coordinate of station p, in metres.
 * @param[in] station_xq     Offset ECEF x-coordinate of station q, in metres.
 * @param[in] station_yp     Offset ECEF y-coordinate of station p, in metres.
 * @param[in] station_yq     Offset ECEF y-coordinate of station q, in metres.
 * @param[in] inv_wavelength Inverse of the wavelength, in metres.
 * @param[in] time_int_sec   Time averaging interval, in seconds.
 * @param[in] gha0_rad       Greenwich Hour Angle of phase centre, in radians.
 * @param[in] dec0_rad       Declination of phase centre, in radians.
 * @param[out] du_dt         Rate of change of baseline u coordinate, in m/s.
 * @param[out] dv_dt         Rate of change of baseline v coordinate, in m/s.
 * @param[out] dw_dt         Rate of change of baseline w coordinate, in m/s.
 */
OSKAR_INLINE
void oskar_evaluate_baseline_derivatives_inline_d(const double station_xp,
        const double station_xq, const double station_yp,
        const double station_yq, const double inv_wavelength,
        const double time_int_sec, const double gha0_rad,
        const double dec0_rad, double* du_dt, double* dv_dt, double* dw_dt)
{
    double xx, yy, rot_angle, temp, sin_HA, cos_HA, sin_Dec, cos_Dec;
#ifdef __CUDACC__
    sincos(gha0_rad, &sin_HA, &cos_HA);
    sincos(dec0_rad, &sin_Dec, &cos_Dec);
#else
    sin_HA = sin(gha0_rad);
    cos_HA = cos(gha0_rad);
    sin_Dec = sin(dec0_rad);
    cos_Dec = cos(dec0_rad);
#endif
    temp = M_PI * inv_wavelength;
    xx = (station_xp - station_xq) * temp;
    yy = (station_yp - station_yq) * temp;
    rot_angle = OMEGA_EARTH * time_int_sec;
    temp = (xx * sin_HA + yy * cos_HA) * rot_angle;
    *du_dt = (xx * cos_HA - yy * sin_HA) * rot_angle;
    *dv_dt = temp * sin_Dec;
    *dw_dt = -temp * cos_Dec;
}

/**
 * @brief
 * Evaluates various per-baseline terms (single precision).
 *
 * @details
 * This function evaluates various per-baseline terms.
 * These values are used for the baseline filter and to simulate bandwidth
 * smearing.
 *
 * @param[in] station_up     Station u-coordinate of station p, in metres.
 * @param[in] station_uq     Station u-coordinate of station q, in metres.
 * @param[in] station_vp     Station v-coordinate of station p, in metres.
 * @param[in] station_vq     Station v-coordinate of station q, in metres.
 * @param[in] station_wp     Station w-coordinate of station p, in metres.
 * @param[in] station_wq     Station w-coordinate of station q, in metres.
 * @param[in] inv_wavelength Inverse of the wavelength, in metres.
 * @param[in] frac_bandwidth Bandwidth divided by frequency.
 * @param[out] uv_len        The uv distance, in wavelengths.
 * @param[out] uu            Modified baseline u-coordinate.
 * @param[out] vv            Modified baseline v-coordinate.
 * @param[out] ww            Modified baseline w-coordinate.
 * @param[out] uu2           Baseline length of u, in wavelengths, squared.
 * @param[out] vv2           Baseline length of v, in wavelengths, squared.
 * @param[out] uuvv          2.0 * u * v, with u and v in wavelengths.
 */
OSKAR_INLINE
void oskar_evaluate_baseline_terms_inline_f(const float station_up,
        const float station_uq, const float station_vp,
        const float station_vq, const float station_wp,
        const float station_wq, const float inv_wavelength,
        const float frac_bandwidth, float* uv_len, float* uu, float* vv,
        float* ww, float* uu2, float* vv2, float* uuvv)
{
    float f;

    /* Baseline distance, in wavelengths. */
    *uu   = (station_up - station_uq) * inv_wavelength;
    *vv   = (station_vp - station_vq) * inv_wavelength;
    *ww   = (station_wp - station_wq) * inv_wavelength;

    /* Quantities needed for evaluating response to a Gaussian source. */
    *uu2  = *uu * *uu;
    *vv2  = *vv * *vv;
    *uv_len = sqrtf(*uu2 + *vv2);
    *uuvv = 2.0f * *uu * *vv;

    /* Modify the baseline distance to include the common components
     * of the bandwidth smearing term. */
    f = M_PIf * frac_bandwidth;
    *uu *= f;
    *vv *= f;
    *ww *= f;
}

/**
 * @brief
 * Evaluates various per-baseline terms (double precision).
 *
 * @details
 * This function evaluates various per-baseline terms.
 * These values are used for the baseline filter and to simulate bandwidth
 * smearing.
 *
 * @param[in] station_up     Station u-coordinate of station p, in metres.
 * @param[in] station_uq     Station u-coordinate of station q, in metres.
 * @param[in] station_vp     Station v-coordinate of station p, in metres.
 * @param[in] station_vq     Station v-coordinate of station q, in metres.
 * @param[in] station_wp     Station w-coordinate of station p, in metres.
 * @param[in] station_wq     Station w-coordinate of station q, in metres.
 * @param[in] inv_wavelength Inverse of the wavelength, in metres.
 * @param[in] frac_bandwidth Bandwidth divided by frequency.
 * @param[out] uv_len        The uv distance, in wavelengths.
 * @param[out] uu            Modified baseline u-coordinate.
 * @param[out] vv            Modified baseline v-coordinate.
 * @param[out] ww            Modified baseline w-coordinate.
 * @param[out] uu2           Baseline length of u, in wavelengths, squared.
 * @param[out] vv2           Baseline length of v, in wavelengths, squared.
 * @param[out] uuvv          2.0 * u * v, with u and v in wavelengths.
 */
OSKAR_INLINE
void oskar_evaluate_baseline_terms_inline_d(const double station_up,
        const double station_uq, const double station_vp,
        const double station_vq, const double station_wp,
        const double station_wq, const double inv_wavelength,
        const double frac_bandwidth, double* uv_len, double* uu, double* vv,
        double* ww, double* uu2, double* vv2, double* uuvv)
{
    double f;

    /* Baseline distance, in wavelengths. */
    *uu   = (station_up - station_uq) * inv_wavelength;
    *vv   = (station_vp - station_vq) * inv_wavelength;
    *ww   = (station_wp - station_wq) * inv_wavelength;

    /* Quantities needed for evaluating response to a Gaussian source. */
    *uu2  = *uu * *uu;
    *vv2  = *vv * *vv;
    *uv_len = sqrt(*uu2 + *vv2);
    *uuvv = 2.0 * *uu * *vv;

    /* Modify the baseline distance to include the common components
     * of the bandwidth smearing term. */
    f = M_PI * frac_bandwidth;
    *uu *= f;
    *vv *= f;
    *ww *= f;
}

/**
 * @brief
 * Evaluates the time-smearing term (single precision).
 *
 * @brief
 * This function evaluates the time-smearing term, given the time derivatives
 * of the baseline coordinates, and the source direction cosines.
 */
OSKAR_INLINE
float oskar_evaluate_time_smearing_f(const float du_dt, const float dv_dt,
        const float dw_dt, const float l, const float m, const float n)
{
    return oskar_sinc_f(du_dt * l + dv_dt * m + dw_dt * (n - 1.0f));
}

/**
 * @brief
 * Evaluates the time-smearing term (double precision).
 *
 * @brief
 * This function evaluates the time-smearing term, given the time derivatives
 * of the baseline coordinates, and the source direction cosines.
 */
OSKAR_INLINE
double oskar_evaluate_time_smearing_d(const double du_dt, const double dv_dt,
        const double dw_dt, const double l, const double m, const double n)
{
    return oskar_sinc_d(du_dt * l + dv_dt * m + dw_dt * (n - 1.0));
}

/**
 * @brief
 * Evaluates the baseline index for the station pair.
 *
 * @details
 * This function evaluates the baseline index, given the indices of the two
 * stations, and the number of stations.
 *
 * @param[in] num_stations  The number of stations.
 * @param[in] p             Index of station p.
 * @param[in] q             Index of station q.
 */
OSKAR_INLINE
int oskar_evaluate_baseline_index_inline(const int num_stations,
        const int p, const int q)
{
    return q * (num_stations - 1) - (q - 1) * q / 2 + p - q - 1;
}

/**
 * @brief
 * Clears a complex matrix (single precision).
 *
 * @details
 * This function clears a complex matrix by setting all its elements to zero.
 *
 * @param[in] m The matrix to clear.
 */
OSKAR_INLINE
void oskar_clear_complex_matrix_f(float4c* m)
{
    m->a.x = 0.0f;
    m->a.y = 0.0f;
    m->b.x = 0.0f;
    m->b.y = 0.0f;
    m->c.x = 0.0f;
    m->c.y = 0.0f;
    m->d.x = 0.0f;
    m->d.y = 0.0f;
}

/**
 * @brief
 * Clears a complex matrix (double precision).
 *
 * @details
 * This function clears a complex matrix by setting all its elements to zero.
 *
 * @param[in] m The matrix to clear.
 */
OSKAR_INLINE
void oskar_clear_complex_matrix_d(double4c* m)
{
    m->a.x = 0.0;
    m->a.y = 0.0;
    m->b.x = 0.0;
    m->b.y = 0.0;
    m->c.x = 0.0;
    m->c.y = 0.0;
    m->d.x = 0.0;
    m->d.y = 0.0;
}

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_PRIVATE_CORRELATE_FUNCTIONS_INLINE_H_ */
