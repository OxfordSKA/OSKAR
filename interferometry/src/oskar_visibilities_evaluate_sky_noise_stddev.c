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


#include "interferometry/oskar_visibilities_evaluate_sky_noise_stddev.h"

#include "interferometry/oskar_Visibilities.h"
#include "interferometry/oskar_visibilities_get_channel_amps.h"

#include "utility/oskar_Mem.h"
#include "utility/oskar_mem_add_gaussian_noise.h"
#include "utility/oskar_mem_init.h"

#include "sky/oskar_evaluate_sky_temperature.h"

#include "station/oskar_evaluate_flux_density.h"
#include "station/oskar_evaluate_effective_area.h"

#include "interferometry/oskar_evaluate_baseline_noise_stddev.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_visibilities_evaluate_sky_noise_stddev(oskar_Visibilities* vis,
        const oskar_TelescopeModel* telescope, double spectral_index)
{
    int i, error, ave_num_elements, total_antennas;
    double *flux_density, *temperature, *effective_area;

    if (vis == NULL || telescope == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Allocate memory for noise standard deviation */
    oskar_mem_init(&vis->sky_noise_stddev, OSKAR_DOUBLE, OSKAR_LOCATION_CPU,
            vis->num_channels, OSKAR_TRUE);

    /* Temporary work arrays */
    temperature = (double*)malloc(vis->num_channels * sizeof(double));
    effective_area = (double*)malloc(vis->num_channels * sizeof(double));
    flux_density = (double*)malloc(vis->num_channels * sizeof(double));

    total_antennas = 0;
    ave_num_elements = 0;
    for (i = 0; i < telescope->num_stations; ++i)
    {
        total_antennas += telescope->station[i].num_elements;
    }
    ave_num_elements = total_antennas / telescope->num_stations;

    error = oskar_evaluate_sky_temperature(temperature, vis->num_channels,
            vis->freq_start_hz, vis->freq_inc_hz, spectral_index);
    if (error) return error;

    error = oskar_evaluate_effective_area(effective_area, vis->num_channels,
            vis->freq_start_hz, vis->freq_inc_hz, ave_num_elements);
    if (error) return error;

    error = oskar_evaluate_flux_density(flux_density, vis->num_channels,
            effective_area, temperature);
    if (error) return error;

    error = oskar_evaluate_baseline_noise_stddev(
            (double*)vis->sky_noise_stddev.data,
            vis->num_channels, flux_density, vis->channel_bandwidth_hz,
            vis->time_inc_seconds);
    if (error) return error;

    free(temperature);
    free(effective_area);
    free(flux_density);

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
