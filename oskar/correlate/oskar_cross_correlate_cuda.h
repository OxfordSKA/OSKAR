/*
 * Copyright (c) 2011-2019, The University of Oxford
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

#ifndef OSKAR_CROSS_CORRELATE_CUDA_H_
#define OSKAR_CROSS_CORRELATE_CUDA_H_

/**
 * @file oskar_cross_correlate_cuda.h
 */

#include <oskar_global.h>
#include <utility/oskar_vector_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * CUDA correlate function for point sources (single precision).
 *
 * @details
 * Forms visibilities on all baselines by correlating Jones matrices for pairs
 * of stations and summing along the source dimension.
 *
 * Note that the station x, y, z coordinates must be in the ECEF frame.
 *
 * Note that all pointers refer to device memory, and must not be dereferenced
 * in host code.
 *
 * @param[in] num_sources    Number of sources.
 * @param[in] num_stations   Number of stations.
 * @param[in] offset_out     Output visibility start offset.
 * @param[in] d_jones        Matrix of Jones matrices to correlate.
 * @param[in] d_I            Source Stokes I values, in Jy.
 * @param[in] d_Q            Source Stokes Q values, in Jy.
 * @param[in] d_U            Source Stokes U values, in Jy.
 * @param[in] d_V            Source Stokes V values, in Jy.
 * @param[in] d_l            Source l-direction cosines from phase centre.
 * @param[in] d_m            Source m-direction cosines from phase centre.
 * @param[in] d_n            Source n-direction cosines from phase centre.
 * @param[in] d_station_u    Station u-coordinates, in metres.
 * @param[in] d_station_v    Station v-coordinates, in metres.
 * @param[in] d_station_w    Station w-coordinates, in metres.
 * @param[in] d_station_x    Station x-coordinates, in metres.
 * @param[in] d_station_y    Station y-coordinates, in metres.
 * @param[in] uv_min_lambda  Minimum allowed UV length, in wavelengths.
 * @param[in] uv_max_lambda  Maximum allowed UV length, in wavelengths.
 * @param[in] inv_wavelength Inverse of the wavelength, in metres.
 * @param[in] frac_bandwidth Bandwidth divided by frequency.
 * @param[in] time_int_sec   Time averaging interval, in seconds.
 * @param[in] gha0_rad       Greenwich Hour Angle of phase centre, in radians.
 * @param[in] dec0_rad       Declination of phase centre, in radians.
 * @param[in,out] d_vis      Modified output complex visibilities.
 */
OSKAR_EXPORT
void oskar_cross_correlate_point_cuda_f(
        int num_sources, int num_stations, int offset_out,
        const float4c* d_jones, const float* d_I, const float* d_Q,
        const float* d_U, const float* d_V,
        const float* d_l, const float* d_m,
        const float* d_n, const float* d_station_u,
        const float* d_station_v, const float* d_station_w,
        const float* d_station_x, const float* d_station_y,
        float uv_min_lambda, float uv_max_lambda, float inv_wavelength,
        float frac_bandwidth, float time_int_sec, float gha0_rad,
        float dec0_rad, float4c* d_vis);

/**
 * @brief
 * CUDA correlate function for point sources (double precision).
 *
 * @details
 * Forms visibilities on all baselines by correlating Jones matrices for pairs
 * of stations and summing along the source dimension.
 *
 * Note that the station x, y, z coordinates must be in the ECEF frame.
 *
 * Note that all pointers refer to device memory, and must not be dereferenced
 * in host code.
 *
 * @param[in] num_sources    Number of sources.
 * @param[in] num_stations   Number of stations.
 * @param[in] offset_out     Output visibility start offset.
 * @param[in] d_jones        Matrix of Jones matrices to correlate.
 * @param[in] d_I            Source Stokes I values, in Jy.
 * @param[in] d_Q            Source Stokes Q values, in Jy.
 * @param[in] d_U            Source Stokes U values, in Jy.
 * @param[in] d_V            Source Stokes V values, in Jy.
 * @param[in] d_l            Source l-direction cosines from phase centre.
 * @param[in] d_m            Source m-direction cosines from phase centre.
 * @param[in] d_n            Source n-direction cosines from phase centre.
 * @param[in] d_station_u    Station u-coordinates, in metres.
 * @param[in] d_station_v    Station v-coordinates, in metres.
 * @param[in] d_station_w    Station w-coordinates, in metres.
 * @param[in] d_station_x    Station x-coordinates, in metres.
 * @param[in] d_station_y    Station y-coordinates, in metres.
 * @param[in] uv_min_lambda  Minimum allowed UV length, in wavelengths.
 * @param[in] uv_max_lambda  Maximum allowed UV length, in wavelengths.
 * @param[in] inv_wavelength Inverse of the wavelength, in metres.
 * @param[in] frac_bandwidth Bandwidth divided by frequency.
 * @param[in] time_int_sec   Time averaging interval, in seconds.
 * @param[in] gha0_rad       Greenwich Hour Angle of phase centre, in radians.
 * @param[in] dec0_rad       Declination of phase centre, in radians.
 * @param[in,out] d_vis      Modified output complex visibilities.
 */
OSKAR_EXPORT
void oskar_cross_correlate_point_cuda_d(
        int num_sources, int num_stations, int offset_out,
        const double4c* d_jones, const double* d_I, const double* d_Q,
        const double* d_U, const double* d_V,
        const double* d_l, const double* d_m,
        const double* d_n, const double* d_station_u,
        const double* d_station_v, const double* d_station_w,
        const double* d_station_x, const double* d_station_y,
        double uv_min_lambda, double uv_max_lambda, double inv_wavelength,
        double frac_bandwidth, double time_int_sec, double gha0_rad,
        double dec0_rad, double4c* d_vis);

/**
 * @brief
 * CUDA correlate function for Gaussian sources (single precision).
 *
 * @details
 * Forms visibilities on all baselines by correlating Jones matrices for pairs
 * of stations and summing along the source dimension.
 *
 * Gaussian parameters a, b, and c are assumed to be evaluated when the
 * sky model is loaded.
 *
 * Note that the station x, y coordinates must be in the ECEF frame.
 *
 * Note that all pointers refer to device memory, and must not be dereferenced
 * in host code.
 *
 * @param[in] num_sources    Number of sources.
 * @param[in] num_stations   Number of stations.
 * @param[in] offset_out     Output visibility start offset.
 * @param[in] d_jones        Matrix of Jones matrices to correlate.
 * @param[in] d_I            Source Stokes I values, in Jy.
 * @param[in] d_Q            Source Stokes Q values, in Jy.
 * @param[in] d_U            Source Stokes U values, in Jy.
 * @param[in] d_V            Source Stokes V values, in Jy.
 * @param[in] d_l            Source l-direction cosines from phase centre.
 * @param[in] d_m            Source m-direction cosines from phase centre.
 * @param[in] d_n            Source n-direction cosines from phase centre.
 * @param[in] d_a            Source Gaussian parameter a.
 * @param[in] d_b            Source Gaussian parameter b.
 * @param[in] d_c            Source Gaussian parameter c.
 * @param[in] d_station_u    Station u-coordinates, in metres.
 * @param[in] d_station_v    Station v-coordinates, in metres.
 * @param[in] d_station_w    Station w-coordinates, in metres.
 * @param[in] d_station_x    Station x-coordinates, in metres.
 * @param[in] d_station_y    Station y-coordinates, in metres.
 * @param[in] uv_min_lambda  Minimum allowed UV length, in wavelengths.
 * @param[in] uv_max_lambda  Maximum allowed UV length, in wavelengths.
 * @param[in] inv_wavelength Inverse of the wavelength, in metres.
 * @param[in] frac_bandwidth Bandwidth divided by frequency.
 * @param[in] time_int_sec   Time averaging interval, in seconds.
 * @param[in] gha0_rad       Greenwich Hour Angle of phase centre, in radians.
 * @param[in] dec0_rad       Declination of phase centre, in radians.
 * @param[in,out] d_vis      Modified output complex visibilities.
 */
OSKAR_EXPORT
void oskar_cross_correlate_gaussian_cuda_f(
        int num_sources, int num_stations, int offset_out,
        const float4c* d_jones, const float* d_I, const float* d_Q,
        const float* d_U, const float* d_V,
        const float* d_l, const float* d_m,
        const float* d_n, const float* d_a,
        const float* d_b, const float* d_c,
        const float* d_station_u, const float* d_station_v,
        const float* d_station_w, const float* d_station_x,
        const float* d_station_y, float uv_min_lambda, float uv_max_lambda,
        float inv_wavelength, float frac_bandwidth, float time_int_sec,
        float gha0_rad, float dec0_rad, float4c* d_vis);

/**
 * @brief
 * CUDA correlate function for Gaussian sources (double precision).
 *
 * @details
 * Forms visibilities on all baselines by correlating Jones matrices for pairs
 * of stations and summing along the source dimension.
 *
 * Gaussian parameters a, b, and c are assumed to be evaluated when the
 * sky model is loaded.
 *
 * Note that the station x, y coordinates must be in the ECEF frame.
 *
 * Note that all pointers refer to device memory, and must not be dereferenced
 * in host code.
 *
 * @param[in] num_sources    Number of sources.
 * @param[in] num_stations   Number of stations.
 * @param[in] offset_out     Output visibility start offset.
 * @param[in] d_jones        Matrix of Jones matrices to correlate.
 * @param[in] d_I            Source Stokes I values, in Jy.
 * @param[in] d_Q            Source Stokes Q values, in Jy.
 * @param[in] d_U            Source Stokes U values, in Jy.
 * @param[in] d_V            Source Stokes V values, in Jy.
 * @param[in] d_l            Source l-direction cosines from phase centre.
 * @param[in] d_m            Source m-direction cosines from phase centre.
 * @param[in] d_n            Source n-direction cosines from phase centre.
 * @param[in] d_a            Source Gaussian parameter a.
 * @param[in] d_b            Source Gaussian parameter b.
 * @param[in] d_c            Source Gaussian parameter c.
 * @param[in] d_station_u    Station u-coordinates, in metres.
 * @param[in] d_station_v    Station v-coordinates, in metres.
 * @param[in] d_station_w    Station w-coordinates, in metres.
 * @param[in] d_station_x    Station x-coordinates, in metres.
 * @param[in] d_station_y    Station y-coordinates, in metres.
 * @param[in] uv_min_lambda  Minimum allowed UV length, in wavelengths.
 * @param[in] uv_max_lambda  Maximum allowed UV length, in wavelengths.
 * @param[in] inv_wavelength Inverse of the wavelength, in metres.
 * @param[in] frac_bandwidth Bandwidth divided by frequency.
 * @param[in] time_int_sec   Time averaging interval, in seconds.
 * @param[in] gha0_rad       Greenwich Hour Angle of phase centre, in radians.
 * @param[in] dec0_rad       Declination of phase centre, in radians.
 * @param[in,out] d_vis      Modified output complex visibilities.
 */
OSKAR_EXPORT
void oskar_cross_correlate_gaussian_cuda_d(
        int num_sources, int num_stations, int offset_out,
        const double4c* d_jones, const double* d_I, const double* d_Q,
        const double* d_U, const double* d_V,
        const double* d_l, const double* d_m,
        const double* d_n, const double* d_a,
        const double* d_b, const double* d_c,
        const double* d_station_u, const double* d_station_v,
        const double* d_station_w, const double* d_station_x,
        const double* d_station_y, double uv_min_lambda, double uv_max_lambda,
        double inv_wavelength, double frac_bandwidth, double time_int_sec,
        double gha0_rad, double dec0_rad, double4c* d_vis);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CROSS_CORRELATE_CUDA_H_ */
