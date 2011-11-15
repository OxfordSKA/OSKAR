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

#include "oskar_global.h"
#include "interferometry/oskar_visibilities_init.h"
#include "interferometry/oskar_Visibilities.h"
#include "utility/oskar_mem_init.h"

#ifdef __cplusplus
extern "C"
#endif
int oskar_visibilities_init(oskar_Visibilities* vis, int amp_type, int location,
        int num_channels, int num_times, int num_baselines)
{
    if (!oskar_Mem::is_complex(amp_type))
        return OSKAR_ERR_BAD_DATA_TYPE;

    if (location != OSKAR_LOCATION_GPU && location != OSKAR_LOCATION_CPU)
        return OSKAR_ERR_BAD_LOCATION;

    // Evaluate the coordinate type.
    int coord_type = oskar_Mem::is_double(amp_type) ? OSKAR_DOUBLE : OSKAR_SINGLE;

    // Set dimensions.
    vis->num_channels  = num_channels;
    vis->num_times     = num_times;
    vis->num_baselines = num_baselines;
    int num_samples    = num_channels * num_times * num_baselines;

    // Initialise memory.
    int err = 0;
    err = oskar_mem_init(&vis->uu_metres, coord_type, location, num_samples, 1);
    if (err) return err;
    err = oskar_mem_init(&vis->vv_metres, coord_type, location, num_samples, 1);
    if (err) return err;
    err = oskar_mem_init(&vis->ww_metres, coord_type, location, num_samples, 1);
    if (err) return err;
    err = oskar_mem_init(&vis->amplitude, amp_type, location, num_samples, 1);
    if (err) return err;

    return 0;
}
