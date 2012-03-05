/*
 * Copyright (c) 2011, The University of Oxford
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

#ifndef OSKAR_CUDAF_HOR_LMN_TO_AZ_EL_H_
#define OSKAR_CUDAF_HOR_LMN_TO_AZ_EL_H_

/**
 * @file oskar_cudaf_hor_lmn_to_az_el.h
 */

#include "oskar_global.h"

/**
 * @brief
 * CUDA device function to convert horizontal direction cosines to az/el
 * (single precision).
 *
 * @details
 * This CUDA device function converts horizontal direction cosines to azimuth
 * and elevation angles.
 *
 * The direction cosines are given with respect to geographic East, North and
 * Up (x,y,z or l,m,n respectively). The azimuth is the angle measured from
 * North through East (from y to x) and the elevation from the xy-plane
 * towards z.
 *
 * @param[in] l    Direction cosine measured with respect to the x axis.
 * @param[in] m    Direction cosine measured with respect to the y axis.
 * @param[in] n    Direction cosine measured with respect to the z axis.
 * @param[out] az  Azimuth angle.
 * @param[out] el  Elevation angle.
 */
__device__
void oskar_cudaf_hor_lmn_to_az_el_f(float l, float m, float n, float* az,
        float* el)
{
    *az = atan2f(l, m);
    *el = atan2f(n, sqrtf(l*l + m*m));
}

/**
 * @brief
 * CUDA device function to convert horizontal direction cosines to az/el
 * (double precision).
 *
 * @details
 * This CUDA device function converts horizontal direction cosines to azimuth
 * and elevation angles.
 *
 * The direction cosines are given with respect to geographic East, North and
 * Up (x,y,z or l,m,n respectively). The azimuth is the angle measured from
 * North through East (from y to x) and the elevation from the xy-plane
 * towards z.
 *
 * @param[in] l    Direction cosine measured with respect to the x axis.
 * @param[in] m    Direction cosine measured with respect to the y axis.
 * @param[in] n    Direction cosine measured with respect to the z axis.
 * @param[out] az  Azimuth angle.
 * @param[out] el  Elevation angle.
 */
__device__
void oskar_cudaf_hor_lmn_to_az_el_d(double l, double m, double n, double* az,
        double* el)
{
    *az = atan2(l, m);
    *el = atan2(n, sqrt(l*l + m*m));
}

#endif /* OSKAR_CUDAF_HOR_LMN_TO_AZ_EL_H_ */
