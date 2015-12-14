/*
 * Copyright (c) 2014, The University of Oxford
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

#ifndef OSKAR_PRIVATE_IMAGE_H_
#define OSKAR_PRIVATE_IMAGE_H_

#include <oskar_mem.h>

struct oskar_Image
{
    oskar_Mem* data;
    oskar_Mem* settings_path;
    int grid_type;   /* NEW v2.4 */
    int coord_frame; /* NEW v2.4 */
    int dimension_order[5];
    int image_type;
    int width;
    int height;
    int num_pols;
    int num_times;
    int num_channels;
    int healpix_nside; /* NEW v2.4 */
    double centre_lon_deg;
    double centre_lat_deg;
    double fov_lon_deg;
    double fov_lat_deg;
    double time_start_mjd_utc;
    double time_inc_sec;
    double freq_start_hz;
    double freq_inc_hz;
};

#ifndef OSKAR_IMAGE_TYPEDEF_
#define OSKAR_IMAGE_TYPEDEF_
typedef struct oskar_Image oskar_Image;
#endif /* OSKAR_IMAGE_TYPEDEF_ */

#endif /* OSKAR_PRIVATE_IMAGE_H_ */
