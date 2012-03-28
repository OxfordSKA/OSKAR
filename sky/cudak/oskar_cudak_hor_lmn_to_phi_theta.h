/*
 * Copyright (c) 2012, The University of Oxford
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

#ifndef OSKAR_CUDAK_HOR_LMN_TO_PHI_THETA_H_
#define OSKAR_CUDAK_HOR_LMN_TO_PHI_THETA_H_

/**
 * @file oskar_cudak_hor_lmn_to_phi_theta.h
 */

#include "oskar_global.h"

/**
 * @brief
 * CUDA kernel to convert horizontal direction cosines to phi/theta
 * (single precision).
 *
 * @details
 * This CUDA kernel converts horizontal direction cosines to co-azimuth
 * and zenith distance.
 *
 * The direction cosines are given with respect to geographic East, North and
 * Up (x,y,z or l,m,n respectively). Phi is the angle measured from
 * East through North (from x to y) and theta from the z axis to the xy-plane.
 *
 * @param[in] num_points  Length of all arrays.
 * @param[in] l           Direction cosine measured with respect to the x axis.
 * @param[in] m           Direction cosine measured with respect to the y axis.
 * @param[in] n           Direction cosine measured with respect to the z axis.
 * @param[out] phi        Co-azimuth angle.
 * @param[out] theta      Zenith distance.
 */
__global__
void oskar_cudak_hor_lmn_to_phi_theta_f(const int num_points,
        const float* l, const float* m, const float* n, float* phi,
        float* theta);

/**
 * @brief
 * CUDA kernel to convert horizontal direction cosines to phi/theta
 * (double precision).
 *
 * @details
 * This CUDA kernel converts horizontal direction cosines to co-azimuth
 * and zenith distance.
 *
 * The direction cosines are given with respect to geographic East, North and
 * Up (x,y,z or l,m,n respectively). Phi is the angle measured from
 * East through North (from x to y) and theta from the z axis to the xy-plane.
 *
 * @param[in] num_points  Length of all arrays.
 * @param[in] l           Direction cosine measured with respect to the x axis.
 * @param[in] m           Direction cosine measured with respect to the y axis.
 * @param[in] n           Direction cosine measured with respect to the z axis.
 * @param[out] phi        Co-azimuth angle.
 * @param[out] theta      Zenith distance.
 */
__global__
void oskar_cudak_hor_lmn_to_phi_theta_d(const int num_points,
        const double* l, const double* m, const double* n, double* phi,
        double* theta);

#endif /* OSKAR_CUDAK_HOR_LMN_TO_PHI_THETA_H_ */
