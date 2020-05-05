/*
 * Copyright (c) 2015-2020, The University of Oxford
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

#include <stdlib.h>
#include "math/oskar_cmath.h"

#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_set_element_feed_angle(oskar_Station* station, int feed,
        int index, double alpha_deg, double beta_deg, double gamma_deg,
        int* status)
{
    const double deg2rad = M_PI / 180.0;
    if (*status || !station) return;
    if (index >= station->num_elements || feed > 1)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return;
    }
    oskar_mem_double(station->element_euler_cpu[feed][0], status)[index] =
            alpha_deg * deg2rad;
    oskar_mem_double(station->element_euler_cpu[feed][1], status)[index] =
            beta_deg * deg2rad;
    oskar_mem_double(station->element_euler_cpu[feed][2], status)[index] =
            gamma_deg * deg2rad;
}

#ifdef __cplusplus
}
#endif
