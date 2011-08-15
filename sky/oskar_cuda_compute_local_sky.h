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

#ifndef OSKAR_CUDA_COMPUTE_LOCAL_SKY_H_
#define OSKAR_CUDA_COMPUTE_LOCAL_SKY_H_

/**
 * @file oskar_cuda_compute_local_sky.h
 */

#include "oskar_windows.h"
#include "sky/oskar_SkyModel.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Computes the local sky model from the global sky model.
 * (single precision).
 *
 * @details
 *
 * @param[in]  hd_global The input global sky model.
 * @param[in]  lst       The current local sidereal time in radians.
 * @param[in]  lat       The geographic latitude of the observer.
 * @param[out] hd_local  The output local sky model.
 */
DllExport
int oskar_cuda_compute_local_sky_f(const oskar_SkyModelGlobal_f* hd_global,
		float lst, float lat, oskar_SkyModelLocal_f* hd_local);

/**
 * @brief
 * Computes the local sky model from the global sky model.
 * (double precision).
 *
 * @details
 *
 * @param[in]  hd_global The input global sky model.
 * @param[in]  lst       The current local sidereal time in radians.
 * @param[in]  lat       The geographic latitude of the observer.
 * @param[out] hd_local  The output local sky model.
 */
DllExport
int oskar_cuda_compute_local_sky_d(const oskar_SkyModelGlobal_d* hd_global,
		double lst, double lat, oskar_SkyModelLocal_d* hd_local);

#ifdef __cplusplus
}
#endif

#endif // OSKAR_CUDA_COMPUTE_LOCAL_SKY_H_
