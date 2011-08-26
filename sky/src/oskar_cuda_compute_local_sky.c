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

#include "sky/oskar_cuda_compute_local_sky.h"
#include "sky/oskar_cuda_horizon_clip.h"
#include "sky/oskar_cuda_transform_to_local_stokes.h"

#ifdef __cplusplus
extern "C" {
#endif

// Single precision.
int oskar_cuda_compute_local_sky_f(const oskar_SkyModelGlobal_f* hd_global,
        float lst, float lat, oskar_SkyModelLocal_f* hd_local)
{
    // Initialise error code.
    int err = 0;

    // Clip sources below the horizon.
    err = oskar_cuda_horizon_clip_f(hd_global, lst, lat, hd_local);
    if (err) return err;

    // Compute the local Stokes parameters.
    err = oskar_cuda_transform_to_local_stokes_f(hd_local->num_sources,
            hd_local->RA, hd_local->Dec, lst, lat, hd_local->Q, hd_local->U);
    return err;
}

// Double precision.
int oskar_cuda_compute_local_sky_d(const oskar_SkyModelGlobal_d* hd_global,
        double lst, double lat, oskar_SkyModelLocal_d* hd_local)
{
    // Initialise error code.
    int err = 0;

    // Clip sources below the horizon.
    err = oskar_cuda_horizon_clip_d(hd_global, lst, lat, hd_local);
    if (err) return err;

    // Compute the local Stokes parameters.
    err = oskar_cuda_transform_to_local_stokes_d(hd_local->num_sources,
            hd_local->RA, hd_local->Dec, lst, lat, hd_local->Q, hd_local->U);
    return err;
}

#ifdef __cplusplus
}
#endif
