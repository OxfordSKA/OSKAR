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


#include "station/oskar_evaluate_station_receiver_noise_stddev.h"
#include "utility/oskar_mem_init.h"
#include <stdlib.h>
#include <math.h>

#ifndef kB
#define kB 1.3806503e-23
#endif

#ifdef __cplusplus
extern "C" {
#endif

int oskar_evaluate_station_receiver_noise_stddev(double* station_noise_stddev,
        const double* receiver_temperature, int num_channels, double bandwidth,
        double integration_time, int num_antennas)
{
    int i;
    double s;

    if (station_noise_stddev == NULL || receiver_temperature == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    for (i = 0; i < num_channels; ++i)
    {
        s = kB * receiver_temperature[i] * 1.0e26;
        station_noise_stddev[i] = s  / sqrt(2.0 * num_antennas *
                bandwidth * integration_time);
    }

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
