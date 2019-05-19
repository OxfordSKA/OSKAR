/*
 * Copyright (c) 2014-2019, The University of Oxford
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

#ifndef OSKAR_CROSS_CORRELATE_SCALAR_OMP_H_
#define OSKAR_CROSS_CORRELATE_SCALAR_OMP_H_

/**
 * @file oskar_cross_correlate_scalar_omp.h
 */

#include <oskar_global.h>
#include <utility/oskar_vector_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Correlate function for point sources, scalar version (single precision).
 *
 * @details
 * Forms visibilities on all baselines by correlating Jones scalars for pairs
 * of stations and summing along the source dimension.
 *
 * Note that the station x, y, z coordinates must be in the ECEF frame.
 *
 * @param[in] num_sources    Number of sources.
 * @param[in] num_stations   Number of stations.
 * @param[in] offset_out     Output visibility start offset.
 * @param[in] jones          Matrix of Jones scalars to correlate.
 * @param[in] I              Source Stokes I values, in Jy.
 * @param[in] l              Source l-direction cosines from phase centre.
 * @param[in] m              Source m-direction cosines from phase centre.
 * @param[in] n              Source n-direction cosines from phase centre.
 * @param[in] station_u      Station u-coordinates, in metres.
 * @param[in] station_v      Station v-coordinates, in metres.
 * @param[in] station_w      Station w-coordinates, in metres.
 * @param[in] station_x      Station x-coordinates, in metres.
 * @param[in] station_y      Station y-coordinates, in metres.
 * @param[in] uv_min_lambda  Minimum allowed UV length, in wavelengths.
 * @param[in] uv_max_lambda  Maximum allowed UV length, in wavelengths.
 * @param[in] inv_wavelength Inverse of the wavelength, in metres.
 * @param[in] frac_bandwidth Bandwidth divided by frequency.
 * @param[in] time_int_sec   Time averaging interval, in seconds.
 * @param[in] gha0_rad       Greenwich Hour Angle of phase centre, in radians.
 * @param[in] dec0_rad       Declination of phase centre, in radians.
 * @param[in,out] vis        Modified output complex visibilities.
 */
OSKAR_EXPORT
void oskar_cross_correlate_scalar_point_omp_f(
        int num_sources, int num_stations, int offset_out,
        const float2* jones, const float* I, const float* l,
        const float* m, const float* n,
        const float* station_u, const float* station_v,
        const float* station_w, const float* station_x,
        const float* station_y, float uv_min_lambda, float uv_max_lambda,
        float inv_wavelength, float frac_bandwidth, const float time_int_sec,
        const float gha0_rad, const float dec0_rad, float2* vis);

/**
 * @brief
 * Correlate function for point sources, scalar version (double precision).
 *
 * @details
 * Forms visibilities on all baselines by correlating Jones scalars for pairs
 * of stations and summing along the source dimension.
 *
 * Note that the station x, y, z coordinates must be in the ECEF frame.
 *
 * @param[in] num_sources    Number of sources.
 * @param[in] num_stations   Number of stations.
 * @param[in] offset_out     Output visibility start offset.
 * @param[in] jones          Matrix of Jones scalars to correlate.
 * @param[in] I              Source Stokes I values, in Jy.
 * @param[in] l              Source l-direction cosines from phase centre.
 * @param[in] m              Source m-direction cosines from phase centre.
 * @param[in] n              Source n-direction cosines from phase centre.
 * @param[in] station_u      Station u-coordinates, in metres.
 * @param[in] station_v      Station v-coordinates, in metres.
 * @param[in] station_w      Station w-coordinates, in metres.
 * @param[in] station_x      Station x-coordinates, in metres.
 * @param[in] station_y      Station y-coordinates, in metres.
 * @param[in] uv_min_lambda  Minimum allowed UV length, in wavelengths.
 * @param[in] uv_max_lambda  Maximum allowed UV length, in wavelengths.
 * @param[in] inv_wavelength Inverse of the wavelength, in metres.
 * @param[in] frac_bandwidth Bandwidth divided by frequency.
 * @param[in] time_int_sec   Time averaging interval, in seconds.
 * @param[in] gha0_rad       Greenwich Hour Angle of phase centre, in radians.
 * @param[in] dec0_rad       Declination of phase centre, in radians.
 * @param[in,out] vis        Modified output complex visibilities.
 */
OSKAR_EXPORT
void oskar_cross_correlate_scalar_point_omp_d(
        int num_sources, int num_stations, int offset_out,
        const double2* jones, const double* I, const double* l,
        const double* m, const double* n,
        const double* station_u, const double* station_v,
        const double* station_w, const double* station_x,
        const double* station_y, double uv_min_lambda, double uv_max_lambda,
        double inv_wavelength, double frac_bandwidth, const double time_int_sec,
        const double gha0_rad, const double dec0_rad, double2* vis);

/**
 * @brief
 * Correlate function for Gaussian sources, scalar version (single precision).
 *
 * @details
 * Forms visibilities on all baselines by correlating Jones scalars for pairs
 * of stations and summing along the source dimension.
 *
 * Gaussian parameters a, b, and c are assumed to be evaluated when the
 * sky model is loaded.
 *
 * Note that the station x, y coordinates must be in the ECEF frame.
 *
 * @param[in] num_sources    Number of sources.
 * @param[in] num_stations   Number of stations.
 * @param[in] offset_out     Output visibility start offset.
 * @param[in] jones          Matrix of Jones scalars to correlate.
 * @param[in] I              Source Stokes I values, in Jy.
 * @param[in] l              Source l-direction cosines from phase centre.
 * @param[in] m              Source m-direction cosines from phase centre.
 * @param[in] n              Source n-direction cosines from phase centre.
 * @param[in] a              Source Gaussian parameter a.
 * @param[in] b              Source Gaussian parameter b.
 * @param[in] c              Source Gaussian parameter c.
 * @param[in] station_u      Station u-coordinates, in metres.
 * @param[in] station_v      Station v-coordinates, in metres.
 * @param[in] station_w      Station w-coordinates, in metres.
 * @param[in] station_x      Station x-coordinates, in metres.
 * @param[in] station_y      Station y-coordinates, in metres.
 * @param[in] uv_min_lambda  Minimum allowed UV length, in wavelengths.
 * @param[in] uv_max_lambda  Maximum allowed UV length, in wavelengths.
 * @param[in] inv_wavelength Inverse of the wavelength, in metres.
 * @param[in] frac_bandwidth Bandwidth divided by frequency.
 * @param[in] time_int_sec   Time averaging interval, in seconds.
 * @param[in] gha0_rad       Greenwich Hour Angle of phase centre, in radians.
 * @param[in] dec0_rad       Declination of phase centre, in radians.
 * @param[in,out] vis        Modified output complex visibilities.
 */
OSKAR_EXPORT
void oskar_cross_correlate_scalar_gaussian_omp_f(
        int num_sources, int num_stations, int offset_out,
        const float2* jones, const float* I, const float* l,
        const float* m, const float* n,
        const float* a, const float* b,
        const float* c, const float* station_u,
        const float* station_v, const float* station_w,
        const float* station_x, const float* station_y,
        float uv_min_lambda, float uv_max_lambda, float inv_wavelength,
        float frac_bandwidth, float time_int_sec, float gha0_rad,
        float dec0_rad, float2* vis);

/**
 * @brief
 * Correlate function for Gaussian sources, scalar version (double precision).
 *
 * @details
 * Forms visibilities on all baselines by correlating Jones scalars for pairs
 * of stations and summing along the source dimension.
 *
 * Gaussian parameters a, b, and c are assumed to be evaluated when the
 * sky model is loaded.
 *
 * Note that the station x, y coordinates must be in the ECEF frame.
 *
 * @param[in] num_sources    Number of sources.
 * @param[in] num_stations   Number of stations.
 * @param[in] offset_out     Output visibility start offset.
 * @param[in] jones          Matrix of Jones scalars to correlate.
 * @param[in] I              Source Stokes I values, in Jy.
 * @param[in] l              Source l-direction cosines from phase centre.
 * @param[in] m              Source m-direction cosines from phase centre.
 * @param[in] n              Source n-direction cosines from phase centre.
 * @param[in] a              Source Gaussian parameter a.
 * @param[in] b              Source Gaussian parameter b.
 * @param[in] c              Source Gaussian parameter c.
 * @param[in] station_u      Station u-coordinates, in metres.
 * @param[in] station_v      Station v-coordinates, in metres.
 * @param[in] station_w      Station w-coordinates, in metres.
 * @param[in] station_x      Station x-coordinates, in metres.
 * @param[in] station_y      Station y-coordinates, in metres.
 * @param[in] uv_min_lambda  Minimum allowed UV length, in wavelengths.
 * @param[in] uv_max_lambda  Maximum allowed UV length, in wavelengths.
 * @param[in] inv_wavelength Inverse of the wavelength, in metres.
 * @param[in] frac_bandwidth Bandwidth divided by frequency.
 * @param[in] time_int_sec   Time averaging interval, in seconds.
 * @param[in] gha0_rad       Greenwich Hour Angle of phase centre, in radians.
 * @param[in] dec0_rad       Declination of phase centre, in radians.
 * @param[in,out] vis        Modified output complex visibilities.
 */
OSKAR_EXPORT
void oskar_cross_correlate_scalar_gaussian_omp_d(
        int num_sources, int num_stations, int offset_out,
        const double2* jones, const double* I, const double* l,
        const double* m, const double* n,
        const double* a, const double* b,
        const double* c, const double* station_u,
        const double* station_v, const double* station_w,
        const double* station_x, const double* station_y,
        double uv_min_lambda, double uv_max_lambda, double inv_wavelength,
        double frac_bandwidth, double time_int_sec, double gha0_rad,
        double dec0_rad, double2* vis);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CROSS_CORRELATE_SCALAR_OMP_H_ */
