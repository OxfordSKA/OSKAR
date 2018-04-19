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

#ifndef OSKAR_PRIVATE_CORRELATE_FUNCTIONS_INLINE_H_
#define OSKAR_PRIVATE_CORRELATE_FUNCTIONS_INLINE_H_

#include <oskar_global.h>
#include <math/oskar_cmath.h>
#include <math/oskar_multiply_inline.h>
#include <math/oskar_kahan_sum.h>

#define OMEGA_EARTH  7.272205217e-5  /* radians/sec */

#ifdef __cplusplus

/**
 * @brief
 * Function to evaluate sinc(x).
 *
 * @details
 * This function evaluates sinc(x) = sin(x) / x.
 *
 * @param[in] x Function argument.
 */
template <typename T>
OSKAR_INLINE
T oskar_sinc(const T x);

template <>
double oskar_sinc<double>(const double x)
{
    return (x == 0.0) ? 1.0 : sin(x) / x;
}

template <>
float oskar_sinc<float>(const float x)
{
    return (x == 0.0f) ? 1.0f : sinf(x) / x;
}

#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 300
    #if CUDART_VERSION >= 9000
        #define WARP_SHUFFLE_XOR(VAR, LANEMASK) __shfl_xor_sync(0xFFFFFFFF, VAR, LANEMASK)
    #else
        #define WARP_SHUFFLE_XOR(VAR, LANEMASK) __shfl_xor(VAR, LANEMASK)
    #endif
#else
    #define WARP_SHUFFLE_XOR(VAR, LANEMASK) 0
#endif

#define OSKAR_WARP_REDUCE(A) {                                             \
        (A) += WARP_SHUFFLE_XOR((A), 1);                                   \
        (A) += WARP_SHUFFLE_XOR((A), 2);                                   \
        (A) += WARP_SHUFFLE_XOR((A), 4);                                   \
        (A) += WARP_SHUFFLE_XOR((A), 8);                                   \
        (A) += WARP_SHUFFLE_XOR((A), 16); }

#endif /* __cplusplus */

#ifdef __cplusplus
extern "C" {
#endif

/* Evaluates sine and cosine of x. */
#ifdef __CUDACC__
#define OSKAR_SINCOS(REAL, X, S, C) sincos((REAL)X, &S, &C)
#else
#define OSKAR_SINCOS(REAL, X, S, C) S = sin((REAL)X); C = cos((REAL)X)
#endif

#define OSKAR_ADD_TO_VIS_POL(V, M)                                         \
        V->a.x += M.a.x; V->a.y += M.a.y;                                  \
        V->b.x += M.b.x; V->b.y += M.b.y;                                  \
        V->c.x += M.c.x; V->c.y += M.c.y;                                  \
        V->d.x += M.d.x; V->d.y += M.d.y;

/* Evaluate various per-baseline terms. */
#define OSKAR_BASELINE_TERMS(REAL, S_UP, S_UQ, S_VP, S_VQ, S_WP, S_WQ, UU, VV, WW, UU2, VV2, UUVV, UV_LEN) { \
        UU   = (S_UP - S_UQ) * inv_wavelength;                             \
        VV   = (S_VP - S_VQ) * inv_wavelength;                             \
        WW   = (S_WP - S_WQ) * inv_wavelength;                             \
        UU2  = UU * UU; VV2  = VV * VV;                                    \
        UV_LEN = sqrt((REAL) (UU2 + VV2));                                 \
        UUVV = 2 * UU * VV;                                                \
        const REAL f = ((REAL) M_PI) * frac_bandwidth;                     \
        UU *= f; VV *= f; WW *= f; }

/* Evaluate baseline deltas for time-average smearing. */
#define OSKAR_BASELINE_DELTAS(REAL, S_XP, S_XQ, S_YP, S_YQ, DU, DV, DW) {  \
        REAL xx, yy, rot_angle, temp, sin_HA, cos_HA, sin_Dec, cos_Dec;    \
        OSKAR_SINCOS(REAL, gha0_rad, sin_HA, cos_HA);                      \
        OSKAR_SINCOS(REAL, dec0_rad, sin_Dec, cos_Dec);                    \
        temp = ((REAL) M_PI) * inv_wavelength;                             \
        xx = (S_XP - S_XQ) * temp;                                         \
        yy = (S_YP - S_YQ) * temp;                                         \
        rot_angle = ((REAL) OMEGA_EARTH) * time_int_sec;                   \
        temp = (xx * sin_HA + yy * cos_HA) * rot_angle;                    \
        DU = (xx * cos_HA - yy * sin_HA) * rot_angle;                      \
        DV = temp * sin_Dec;                                               \
        DW = -temp * cos_Dec; }

/* Clears a complex matrix. */
#define OSKAR_CLEAR_COMPLEX_MATRIX(REAL, M) {                              \
        M.a.x = M.a.y = M.b.x = M.b.y = (REAL)0;                           \
        M.c.x = M.c.y = M.d.x = M.d.y = (REAL)0; }

/* Construct source brightness matrix (ignoring c, as it's Hermitian). */
#define OSKAR_CONSTRUCT_B(REAL, B, SRC_I, SRC_Q, SRC_U, SRC_V) {           \
        REAL s_I__, s_Q__; s_I__ = SRC_I; s_Q__ = SRC_Q;                   \
        B.b.x = SRC_U; B.b.y = SRC_V;                                      \
        B.a.x = s_I__ + s_Q__; B.d.x = s_I__ - s_Q__; }

#if defined(__CUDACC__) && __CUDA_ARCH__ >= 350
/* Uses __ldg() instruction for global load via texture cache. */
/* Available only from CUDA architecture 3.5. */
#define OSKAR_LOAD_MATRIX(M, IND8) {                                       \
        M.a = __ldg(&(IND8.a));                                            \
        M.b = __ldg(&(IND8.b));                                            \
        M.c = __ldg(&(IND8.c));                                            \
        M.d = __ldg(&(IND8.d)); }
#else
#define OSKAR_LOAD_MATRIX(M, IND8) M = IND8;
#endif

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
    OSKAR_CONSTRUCT_B(float, m2, I[source_id], Q[source_id], U[source_id], V[source_id])

    /* Multiply first Jones matrix with source brightness matrix. */
    OSKAR_LOAD_MATRIX(m1, J[source_id])
    oskar_multiply_complex_matrix_hermitian_in_place_f(&m1, &m2);

    /* Multiply result with second (Hermitian transposed) Jones matrix. */
    OSKAR_LOAD_MATRIX(m2, J[source_id])
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
    OSKAR_CONSTRUCT_B(double, m2, I[source_id], Q[source_id], U[source_id], V[source_id])

    /* Multiply first Jones matrix with source brightness matrix. */
    OSKAR_LOAD_MATRIX(m1, J[source_id])
    oskar_multiply_complex_matrix_hermitian_in_place_d(&m1, &m2);

    /* Multiply result with second (Hermitian transposed) Jones matrix. */
    OSKAR_LOAD_MATRIX(m2, J[source_id])
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

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_PRIVATE_CORRELATE_FUNCTIONS_INLINE_H_ */
