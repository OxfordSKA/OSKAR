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

#ifndef OSKAR_CORRELATE_FUNCTIONS_INLINE_H_
#define OSKAR_CORRELATE_FUNCTIONS_INLINE_H_

/**
 * @file oskar_correlate_functions_inline.h
 */

#include <oskar_global.h>
#include <oskar_multiply_inline.h>
#include <oskar_kahan_sum.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#ifndef M_PIf
#define M_PIf 3.14159265358979323846264338327950288f
#endif

#define OMEGA_EARTH  7.272205217e-5  /* radians/sec */
#define OMEGA_EARTHf 7.272205217e-5f /* radians/sec */

#ifdef __cplusplus
extern "C" {
#endif

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
 * The output visibility on the baseline is updated according to:
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
 * @param[in,out] V_pq  Running total of source visibilities on the baseline.
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

    /* Construct source brightness matrix (ignoring c, as it's Hermitian). */
    {
        float s_I, s_Q;
        s_I = I[source_id];
        s_Q = Q[source_id];
        m2.b.x = U[source_id];
        m2.b.y = V[source_id];
        m2.a.x = s_I + s_Q;
        m2.d.x = s_I - s_Q;
    }

#if defined(__CUDACC__) && __CUDA_ARCH__ >= 350
    /* Uses __ldg() instruction for global load via texture cache. */
    /* Available only from CUDA architecture 3.5. */
    
    /* Multiply first Jones matrix with source brightness matrix. */
    m1.a = __ldg(&(J_p[source_id].a));
    m1.b = __ldg(&(J_p[source_id].b));
    m1.c = __ldg(&(J_p[source_id].c));
    m1.d = __ldg(&(J_p[source_id].d));
    oskar_multiply_complex_matrix_hermitian_in_place_f(&m1, &m2);

    /* Multiply result with second (Hermitian transposed) Jones matrix. */
    m2.a = __ldg(&(J_q[source_id].a));
    m2.b = __ldg(&(J_q[source_id].b));
    m2.c = __ldg(&(J_q[source_id].c));
    m2.d = __ldg(&(J_q[source_id].d));
    oskar_multiply_complex_matrix_conjugate_transpose_in_place_f(&m1, &m2);
#else
    /* Multiply first Jones matrix with source brightness matrix. */
    m1 = J_p[source_id];
    oskar_multiply_complex_matrix_hermitian_in_place_f(&m1, &m2);

    /* Multiply result with second (Hermitian transposed) Jones matrix. */
    m2 = J_q[source_id];
    oskar_multiply_complex_matrix_conjugate_transpose_in_place_f(&m1, &m2);
#endif /* __CUDA_ARCH__ >= 350 */

#ifdef __CUDACC__
    /* Multiply result by smearing term and accumulate. */
    V_pq->a.x += m1.a.x * smear;
    V_pq->a.y += m1.a.y * smear;
    V_pq->b.x += m1.b.x * smear;
    V_pq->b.y += m1.b.y * smear;
    V_pq->c.x += m1.c.x * smear;
    V_pq->c.y += m1.c.y * smear;
    V_pq->d.x += m1.d.x * smear;
    V_pq->d.y += m1.d.y * smear;
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
 * The output visibility on the baseline is updated according to:
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
 * @param[in,out] V_pq  Running total of source visibilities on the baseline.
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

    /* Construct source brightness matrix (ignoring c, as it's Hermitian). */
    {
        double s_I, s_Q;
        s_I = I[source_id];
        s_Q = Q[source_id];
        m2.b.x = U[source_id];
        m2.b.y = V[source_id];
        m2.a.x = s_I + s_Q;
        m2.d.x = s_I - s_Q;
    }

#if defined(__CUDACC__) && __CUDA_ARCH__ >= 350
    /* Uses __ldg() instruction for global load via texture cache. */
    /* Available only from CUDA architecture 3.5. */

    /* Multiply first Jones matrix with source brightness matrix. */
    m1.a = __ldg(&(J_p[source_id].a));
    m1.b = __ldg(&(J_p[source_id].b));
    m1.c = __ldg(&(J_p[source_id].c));
    m1.d = __ldg(&(J_p[source_id].d));
    oskar_multiply_complex_matrix_hermitian_in_place_d(&m1, &m2);

    /* Multiply result with second (Hermitian transposed) Jones matrix. */
    m2.a = __ldg(&(J_q[source_id].a));
    m2.b = __ldg(&(J_q[source_id].b));
    m2.c = __ldg(&(J_q[source_id].c));
    m2.d = __ldg(&(J_q[source_id].d));
    oskar_multiply_complex_matrix_conjugate_transpose_in_place_d(&m1, &m2);
#else
    /* Multiply first Jones matrix with source brightness matrix. */
    m1 = J_p[source_id];
    oskar_multiply_complex_matrix_hermitian_in_place_d(&m1, &m2);

    /* Multiply result with second (Hermitian transposed) Jones matrix. */
    m2 = J_q[source_id];
    oskar_multiply_complex_matrix_conjugate_transpose_in_place_d(&m1, &m2);
#endif

    /* Multiply result by smearing term and accumulate. */
    V_pq->a.x += m1.a.x * smear;
    V_pq->a.y += m1.a.y * smear;
    V_pq->b.x += m1.b.x * smear;
    V_pq->b.y += m1.b.y * smear;
    V_pq->c.x += m1.c.x * smear;
    V_pq->c.y += m1.c.y * smear;
    V_pq->d.x += m1.d.x * smear;
    V_pq->d.y += m1.d.y * smear;
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
 * Evaluates the baseline length modified by components of the bandwidth
 * smearing term (single precision).
 *
 * @details
 * This function evaluates the baseline length in wavelengths, multiplied by
 * pi and the fractional bandwidth.
 *
 * These values are used to simulate bandwidth smearing.
 *
 * @param[in] station_up     Station u-coordinate of station p, in metres.
 * @param[in] station_uq     Station u-coordinate of station q, in metres.
 * @param[in] station_vp     Station v-coordinate of station p, in metres.
 * @param[in] station_vq     Station v-coordinate of station q, in metres.
 * @param[in] inv_wavelength Inverse of the wavelength, in metres.
 * @param[in] frac_bandwidth Bandwidth divided by frequency.
 * @param[out] uu            Modified baseline u-coordinate.
 * @param[out] vv            Modified baseline u-coordinate.
 */
OSKAR_INLINE
void oskar_evaluate_modified_baseline_inline_f(const float station_up,
        const float station_uq, const float station_vp,
        const float station_vq, const float inv_wavelength,
        const float frac_bandwidth, float* uu, float* vv)
{
    float f;

    /* Get baseline uv-distances modified by bandwidth smearing parameters. */
    f   = M_PIf * frac_bandwidth * inv_wavelength;
    *uu = (station_up - station_uq) * f;
    *vv = (station_vp - station_vq) * f;
}

/**
 * @brief
 * Evaluates the baseline length modified by components of the bandwidth
 * smearing term (single precision).
 *
 * @details
 * This function evaluates the baseline length in wavelengths, multiplied by
 * pi and the fractional bandwidth, and also baseline-dependent values used
 * to simulate the response to Gaussian sources.
 *
 * These values are used to simulate bandwidth smearing.
 *
 * @param[in] station_up     Station u-coordinate of station p, in metres.
 * @param[in] station_uq     Station u-coordinate of station q, in metres.
 * @param[in] station_vp     Station v-coordinate of station p, in metres.
 * @param[in] station_vq     Station v-coordinate of station q, in metres.
 * @param[in] inv_wavelength Inverse of the wavelength, in metres.
 * @param[in] frac_bandwidth Bandwidth divided by frequency.
 * @param[out] uu            Modified baseline u-coordinate.
 * @param[out] vv            Modified baseline u-coordinate.
 * @param[out] uu2           Baseline length of u, in wavelengths, squared.
 * @param[out] vv2           Baseline length of v, in wavelengths, squared.
 * @param[out] uuvv          2.0 * u * v, with u and v in wavelengths.
 */
OSKAR_INLINE
void oskar_evaluate_modified_baseline_gaussian_inline_f(const float station_up,
        const float station_uq, const float station_vp,
        const float station_vq, const float inv_wavelength,
        const float frac_bandwidth, float* uu, float* vv, float* uu2,
        float* vv2, float* uuvv)
{
    float f;

    /* Baseline UV-distance, in wavelengths. */
    *uu   = (station_up - station_uq) * inv_wavelength;
    *vv   = (station_vp - station_vq) * inv_wavelength;

    /* Quantities needed for evaluating response to a Gaussian source. */
    *uu2  = *uu * *uu;
    *vv2  = *vv * *vv;
    *uuvv = 2.0f * *uu * *vv;

    /* Modify the baseline UV-distance to include the common components
     * of the bandwidth smearing term. */
    f = M_PIf * frac_bandwidth;
    *uu *= f;
    *vv *= f;
}

/**
 * @brief
 * Evaluates the baseline length modified by components of the bandwidth
 * smearing term (double precision).
 *
 * @details
 * This function evaluates the baseline length in wavelengths, multiplied by
 * pi and the fractional bandwidth.
 *
 * These values are used to simulate bandwidth smearing.
 *
 * @param[in] station_up     Station u-coordinate of station p, in metres.
 * @param[in] station_uq     Station u-coordinate of station q, in metres.
 * @param[in] station_vp     Station v-coordinate of station p, in metres.
 * @param[in] station_vq     Station v-coordinate of station q, in metres.
 * @param[in] inv_wavelength Inverse of the wavelength, in metres.
 * @param[in] frac_bandwidth Bandwidth divided by frequency.
 * @param[out] uu            Modified baseline u-coordinate.
 * @param[out] vv            Modified baseline u-coordinate.
 */
OSKAR_INLINE
void oskar_evaluate_modified_baseline_inline_d(const double station_up,
        const double station_uq, const double station_vp,
        const double station_vq, const double inv_wavelength,
        const double frac_bandwidth, double* uu, double* vv)
{
    double f;

    /* Get baseline uv-distances modified by bandwidth smearing parameters. */
    f   = M_PI * frac_bandwidth * inv_wavelength;
    *uu = (station_up - station_uq) * f;
    *vv = (station_vp - station_vq) * f;
}

/**
 * @brief
 * Evaluates the baseline length modified by components of the bandwidth
 * smearing term (double precision).
 *
 * @details
 * This function evaluates the baseline length in wavelengths, multiplied by
 * pi and the fractional bandwidth, and also baseline-dependent values used
 * to simulate the response to Gaussian sources.
 *
 * These values are used to simulate bandwidth smearing.
 *
 * @param[in] station_up     Station u-coordinate of station p, in metres.
 * @param[in] station_uq     Station u-coordinate of station q, in metres.
 * @param[in] station_vp     Station v-coordinate of station p, in metres.
 * @param[in] station_vq     Station v-coordinate of station q, in metres.
 * @param[in] inv_wavelength Inverse of the wavelength, in metres.
 * @param[in] frac_bandwidth Bandwidth divided by frequency.
 * @param[out] uu            Modified baseline u-coordinate.
 * @param[out] vv            Modified baseline u-coordinate.
 * @param[out] uu2           Baseline length of u, in wavelengths, squared.
 * @param[out] vv2           Baseline length of v, in wavelengths, squared.
 * @param[out] uuvv          2.0 * u * v, with u and v in wavelengths.
 */
OSKAR_INLINE
void oskar_evaluate_modified_baseline_gaussian_inline_d(const double station_up,
        const double station_uq, const double station_vp,
        const double station_vq, const double inv_wavelength,
        const double frac_bandwidth, double* uu, double* vv, double* uu2,
        double* vv2, double* uuvv)
{
    double f;

    /* Baseline UV-distance, in wavelengths. */
    *uu   = (station_up - station_uq) * inv_wavelength;
    *vv   = (station_vp - station_vq) * inv_wavelength;

    /* Quantities needed for evaluating response to a Gaussian source. */
    *uu2  = *uu * *uu;
    *vv2  = *vv * *vv;
    *uuvv = 2.0 * *uu * *vv;

    /* Modify the baseline UV-distance to include the common components
     * of the bandwidth smearing term. */
    f = M_PI * frac_bandwidth;
    *uu *= f;
    *vv *= f;
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

#endif /* OSKAR_CORRELATE_FUNCTIONS_INLINE_H_ */
