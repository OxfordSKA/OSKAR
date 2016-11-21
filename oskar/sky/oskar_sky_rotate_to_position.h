/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#ifndef OSKAR_SKY_ROTATE_TO_POSITION_H_
#define OSKAR_SKY_ROTATE_TO_POSITION_H_

/**
 * @file oskar_sky_rotate_to_position.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Rotates sources in the sky model to a point in the sky.
 *
 * @details
 * This function can be used to rotate the sky model to the observation phase
 * centre.
 *
 * The rotation matrix is given by:
 *
 *   [ cos(a)sin(d)   -sin(a)   cos(a)cos(d) ]
 *   [ sin(a)sin(d)    cos(a)   sin(a)cos(d) ]
 *   [    -cos(d)        0          sin(d)   ]
 *
 * where a = ra0, d = dec0.
 * This corresponds to a rotation of a around z,
 * followed by a rotation of (90-d) around y.
 *
 * @param[out] sky  Pointer to sky model structure.
 * @param[in] ra0   Right ascension of position.
 * @param[in] dec0  Declination of position.
 * @param[in,out] status Status return code.
 */
OSKAR_EXPORT
void oskar_sky_rotate_to_position(oskar_Sky* sky,
        double ra0, double dec0, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SKY_ROTATE_TO_POSITION_H_ */
