/*
 * Copyright (c) 2013-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_CROSS_CORRELATE_OMP_H_
#define OSKAR_CROSS_CORRELATE_OMP_H_

/**
 * @file oskar_cross_correlate_omp.h
 */

#include <oskar_global.h>
#include <utility/oskar_vector_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Correlate function for point sources (single precision).
 *
 * @details
 * Forms visibilities on all baselines by correlating Jones matrices for pairs
 * of stations and summing along the source dimension.
 *
 * Note that the station x, y, z coordinates must be in the ECEF frame.
 *
 * @param[in] use_casa_phase_convention If set, use the CASA phase convention.
 * @param[in] num_sources    Number of sources.
 * @param[in] num_stations   Number of stations.
 * @param[in] offset_out     Output visibility start offset.
 * @param[in] jones          Matrix of Jones matrices to correlate.
 * @param[in] I              Source Stokes I values, in Jy.
 * @param[in] Q              Source Stokes Q values, in Jy.
 * @param[in] U              Source Stokes U values, in Jy.
 * @param[in] V              Source Stokes V values, in Jy.
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
void oskar_cross_correlate_point_omp_f(
        int use_casa_phase_convention,
        int num_sources, int num_stations, int offset_out,
        const float4c* jones, const float* I, const float* Q,
        const float* U, const float* V,
        const float* l, const float* m, const float* n,
        const float* station_u, const float* station_v,
        const float* station_w,
        const float* station_x, const float* station_y,
        float uv_min_lambda, float uv_max_lambda, float inv_wavelength,
        float frac_bandwidth, float time_int_sec, float gha0_rad,
        float dec0_rad, float4c* vis);

/**
 * @brief
 * Correlate function for point sources (double precision).
 *
 * @details
 * Forms visibilities on all baselines by correlating Jones matrices for pairs
 * of stations and summing along the source dimension.
 *
 * Note that the station x, y, z coordinates must be in the ECEF frame.
 *
 * @param[in] use_casa_phase_convention If set, use the CASA phase convention.
 * @param[in] num_sources    Number of sources.
 * @param[in] num_stations   Number of stations.
 * @param[in] offset_out     Output visibility start offset.
 * @param[in] jones          Matrix of Jones matrices to correlate.
 * @param[in] I              Source Stokes I values, in Jy.
 * @param[in] Q              Source Stokes Q values, in Jy.
 * @param[in] U              Source Stokes U values, in Jy.
 * @param[in] V              Source Stokes V values, in Jy.
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
void oskar_cross_correlate_point_omp_d(
        int use_casa_phase_convention,
        int num_sources, int num_stations, int offset_out,
        const double4c* jones, const double* I, const double* Q,
        const double* U, const double* V,
        const double* l, const double* m, const double* n,
        const double* station_u, const double* station_v,
        const double* station_w,
        const double* station_x, const double* station_y,
        double uv_min_lambda, double uv_max_lambda, double inv_wavelength,
        double frac_bandwidth, double time_int_sec, double gha0_rad,
        double dec0_rad, double4c* vis);

/**
 * @brief
 * Correlate function for Gaussian sources (single precision).
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
 * @param[in] use_casa_phase_convention If set, use the CASA phase convention.
 * @param[in] num_sources    Number of sources.
 * @param[in] num_stations   Number of stations.
 * @param[in] offset_out     Output visibility start offset.
 * @param[in] jones          Matrix of Jones matrices to correlate.
 * @param[in] I              Source Stokes I values, in Jy.
 * @param[in] Q              Source Stokes Q values, in Jy.
 * @param[in] U              Source Stokes U values, in Jy.
 * @param[in] V              Source Stokes V values, in Jy.
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
void oskar_cross_correlate_gaussian_omp_f(
        int use_casa_phase_convention,
        int num_sources, int num_stations, int offset_out,
        const float4c* jones, const float* I, const float* Q,
        const float* U, const float* V,
        const float* l, const float* m, const float* n,
        const float* a, const float* b, const float* c,
        const float* station_u, const float* station_v,
        const float* station_w, const float* station_x,
        const float* station_y, float uv_min_lambda, float uv_max_lambda,
        float inv_wavelength, float frac_bandwidth, float time_int_sec,
        float gha0_rad, float dec0_rad, float4c* vis);

/**
 * @brief
 * Correlate function for Gaussian sources (double precision).
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
 * @param[in] use_casa_phase_convention If set, use the CASA phase convention.
 * @param[in] num_sources    Number of sources.
 * @param[in] num_stations   Number of stations.
 * @param[in] offset_out     Output visibility start offset.
 * @param[in] jones          Matrix of Jones matrices to correlate.
 * @param[in] I              Source Stokes I values, in Jy.
 * @param[in] Q              Source Stokes Q values, in Jy.
 * @param[in] U              Source Stokes U values, in Jy.
 * @param[in] V              Source Stokes V values, in Jy.
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
void oskar_cross_correlate_gaussian_omp_d(
        int use_casa_phase_convention,
        int num_sources, int num_stations, int offset_out,
        const double4c* jones, const double* I, const double* Q,
        const double* U, const double* V,
        const double* l, const double* m, const double* n,
        const double* a, const double* b, const double* c,
        const double* station_u, const double* station_v,
        const double* station_w, const double* station_x,
        const double* station_y, double uv_min_lambda, double uv_max_lambda,
        double inv_wavelength, double frac_bandwidth, double time_int_sec,
        double gha0_rad, double dec0_rad, double4c* vis);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CROSS_CORRELATE_OMP_H_ */
