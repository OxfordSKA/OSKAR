/*
 * Copyright (c) 2013-2019, The University of Oxford
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

#ifndef OSKAR_EVALUATE_IMAGE_LON_LAT_GRID_H_
#define OSKAR_EVALUATE_IMAGE_LON_LAT_GRID_H_

/**
 * @file oskar_evaluate_image_lon_lat_grid.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Generate longitude and latitude coordinates corresponding to image pixel
 * positions.
 *
 * @details
 * This function generates the longitude and latitude coordinates that
 * correspond to image pixel positions.
 *
 * This assumes an orthographic projection.
 *
 * @param[in]  num_pixels_l Image side length in the l-dimension, in pixels.
 * @param[in]  num_pixels_m Image side length in the m-dimension, in pixels.
 * @param[in]  fov_rad_lon  Field-of-view in longitude (l-dimension), in radians.
 * @param[in]  fov_rad_lat  Field-of-view in latitude (m-dimension), in radians.
 * @param[in]  lon_rad      Image centre longitude, in radians.
 * @param[in]  lat_rad      Image centre latitude, in radians.
 * @param[out] lon          Output pixel positions in longitude, in radians.
 * @param[out] lat          Output pixel positions in latitude, in radians.
 * @param[in,out]  status   Status return code.
 */
OSKAR_EXPORT
void oskar_evaluate_image_lon_lat_grid(int num_pixels_l, int num_pixels_m,
        double fov_rad_lon, double fov_rad_lat, double lon_rad, double lat_rad,
        oskar_Mem* lon, oskar_Mem* lat, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_IMAGE_LON_LAT_GRID_H_ */
