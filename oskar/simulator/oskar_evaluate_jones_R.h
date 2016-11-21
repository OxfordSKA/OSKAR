/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#ifndef OSKAR_EVALUATE_JONES_R_H_
#define OSKAR_EVALUATE_JONES_R_H_

/**
 * @file oskar_evaluate_jones_R.h
 */

#include <oskar_global.h>
#include <oskar_sky.h>
#include <oskar_telescope.h>
#include <oskar_jones.h>
#include <oskar_vector_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Function to construct matrices for parallactic angle rotation
 * (single precision).
 *
 * @details
 * This function constructs a set of Jones matrices that will transform the
 * equatorial Stokes parameters into the local horizontal frame of each station.
 * This corresponds to a rotation by the parallactic angle (q) for each source
 * and station. The Jones matrix is:
 *
 * ( cos(q)  -sin(q) )
 * ( sin(q)   cos(q) )
 *
 * @param[out] jones       Output set of Jones matrices.
 * @param[in] num_sources  Number of source positions.
 * @param[in] ra_rad       Source Right Ascension coordinates, in radians.
 * @param[in] dec_rad      Source Declination coordinates, in radians.
 * @param[in] latitude_rad The observer's latitude, in radians.
 * @param[in] lst_rad      The Local Apparent Sidereal Time, in radians.
 */
OSKAR_EXPORT
void oskar_evaluate_jones_R_f(float4c* jones, int num_sources,
        const float* ra, const float* dec, float latitude_rad,
        float lst_rad);

/**
 * @brief
 * Function to construct matrices for parallactic angle rotation
 * (double precision).
 *
 * @details
 * This function constructs a set of Jones matrices that will transform the
 * equatorial Stokes parameters into the local horizontal frame of each station.
 * This corresponds to a rotation by the parallactic angle (q) for each source
 * and station. The Jones matrix is:
 *
 * ( cos(q)  -sin(q) )
 * ( sin(q)   cos(q) )
 *
 * @param[out] jones       Output set of Jones matrices.
 * @param[in] num_sources  Number of source positions.
 * @param[in] ra_rad       Source Right Ascension coordinates, in radians.
 * @param[in] dec_rad      Source Declination coordinates, in radians.
 * @param[in] latitude_rad The observer's latitude, in radians.
 * @param[in] lst_rad      The Local Apparent Sidereal Time, in radians.
 */
OSKAR_EXPORT
void oskar_evaluate_jones_R_d(double4c* jones, int num_sources,
        const double* ra_rad, const double* dec_rad, double latitude_rad,
        double lst_rad);

/**
 * @brief
 * Function to construct matrices for parallactic angle rotation and
 * conversion of linear Stokes parameters from equatorial to local horizontal
 * frame.
 *
 * @details
 * This function constructs a set of Jones matrices that will transform the
 * equatorial Stokes parameters into the local horizontal frame of each station.
 * This corresponds to a rotation by the parallactic angle (q) for each source
 * and station. The Jones matrix is:
 *
 * ( cos(q)  -sin(q) )
 * ( sin(q)   cos(q) )
 *
 * @param[out] R          Output set of Jones matrices.
 * @param[in] num_sources Number of sources to use from coordinate arrays.
 * @param[in] ra_rad      Input Right Ascension values, in radians.
 * @param[in] dec_rad     Input Declination values, in radians.
 * @param[in] telescope   Input telescope model.
 * @param[in] gast        The Greenwich Apparent Sidereal Time, in radians.
 * @param[in,out] status  Status return code.
 */
OSKAR_EXPORT
void oskar_evaluate_jones_R(oskar_Jones* R, int num_sources,
        const oskar_Mem* ra_rad, const oskar_Mem* dec_rad,
        const oskar_Telescope* telescope, double gast, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_JONES_R_H_ */
