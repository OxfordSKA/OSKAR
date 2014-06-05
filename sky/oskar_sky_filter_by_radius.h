/*
 * Copyright (c) 2011-2014, The University of Oxford
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

#ifndef OSKAR_SKY_FILTER_BY_RADIUS_H_
#define OSKAR_SKY_FILTER_BY_RADIUS_H_

/**
 * @file oskar_sky_filter_by_radius.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Removes sources outside given limits.
 *
 * @details
 * This function removes sources from a sky model that lie within
 * the inner radius or beyond the outer radius.
 *
 * @param[in,out] sky          Pointer to sky model.
 * @param[in] inner_radius_rad Inner radius in radians.
 * @param[in] outer_radius_rad Outer radius in radians.
 * @param[in] ra0_rad          Right ascension of the phase centre in radians.
 * @param[in] dec0_rad         Declination of the phase centre in radians.
 * @param[in,out] status       Status return code.
 */
OSKAR_EXPORT
void oskar_sky_filter_by_radius(oskar_Sky* sky, double inner_radius_rad,
        double outer_radius_rad, double ra0_rad, double dec0_rad, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SKY_FILTER_BY_RADIUS_H_ */
