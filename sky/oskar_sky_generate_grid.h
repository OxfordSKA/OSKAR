/*
 * Copyright (c) 2016, The University of Oxford
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

#ifndef OSKAR_SKY_GENERATE_GRID_H_
#define OSKAR_SKY_GENERATE_GRID_H_

/**
 * @file oskar_sky_generate_grid.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Generates a grid of sources at the specified tangent point on the sky.
 *
 * @details
 * Generates a grid of sources at the specified tangent point on the sky.
 *
 * @param[in] precision     Precision of the sky model to create.
 * @param[in] ra0_rad       Right Ascension of grid centre, in radians.
 * @param[in] dec0_rad      Declination of grid centre, in radians.
 * @param[in] side_length   Side length of generated grid.
 * @param[in] fov_rad       Grid field-of-view, in radians.
 * @param[in] mean_flux_jy  Mean Stokes-I source flux, in Jy.
 * @param[in] std_flux_jy   Standard deviation Stokes-I source flux, in Jy.
 * @param[in] seed          Random generator seed.
 * @param[in,out]  status   Status return code.
 */
OSKAR_EXPORT
oskar_Sky* oskar_sky_generate_grid(int precision, double ra0_rad,
        double dec0_rad, int side_length, double fov_rad, double mean_flux_jy,
        double std_flux_jy, int seed, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SKY_GENERATE_GRID_H_ */
