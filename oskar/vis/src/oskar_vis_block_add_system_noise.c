/*
 * Copyright (c) 2015-2016, The University of Oxford
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

#include "vis/private_vis_block.h"
#include "vis/oskar_vis_block.h"
#include "math/oskar_random_gaussian.h"
#include "math/oskar_find_closest_match.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

static void oskar_get_station_std_dev_for_channel(oskar_Mem* station_std_dev,
        double frequency_hz, const oskar_Telescope* tel, int* status)
{
    int i, j, num_stations;
    const oskar_Mem *noise_freq, *noise_rms;
    const oskar_Station *station;

    /* Ensure output array is big enough. */
    num_stations = oskar_telescope_num_stations(tel);
    if ((int)oskar_mem_length(station_std_dev) < num_stations)
        oskar_mem_realloc(station_std_dev, num_stations, status);

    /* Loop over stations and get noise value standard deviation for each. */
    for (i = 0; i < num_stations; ++i)
    {
        station = oskar_telescope_station_const(tel, i);
        noise_freq = oskar_station_noise_freq_hz_const(station);
        noise_rms = oskar_station_noise_rms_jy_const(station);
        j = oskar_find_closest_match(frequency_hz, noise_freq, status);
        oskar_mem_copy_contents(station_std_dev, noise_rms, i, j, 1, status);
    }
}

/* Applies noise to data in a visibility block, for the given channel. */
static void oskar_vis_block_apply_noise(oskar_VisBlock* vis,
        const oskar_Mem* station_std_dev, unsigned int seed,
        unsigned int block_idx, unsigned int channel_idx,
        double channel_bandwidth_hz, double time_int_sec, int* status)
{
    int a1, a2, have_autocorr, have_crosscorr, block_start, b, c = 0, t;
    int num_baselines, num_channels, num_stations, num_times;
    void *acorr_ptr, *xcorr_ptr;
    double rnd[8], std, mean, sefd_conversion;
    const double inv_sqrt2 = 1.0 / sqrt(2.0);

    /* Get pointer to start of block, and block dimensions. */
    have_autocorr  = oskar_vis_block_has_auto_correlations(vis);
    have_crosscorr = oskar_vis_block_has_cross_correlations(vis);
    acorr_ptr      = oskar_mem_void(oskar_vis_block_auto_correlations(vis));
    xcorr_ptr      = oskar_mem_void(oskar_vis_block_cross_correlations(vis));
    num_baselines  = oskar_vis_block_num_baselines(vis);
    num_channels   = oskar_vis_block_num_channels(vis);
    num_stations   = oskar_vis_block_num_stations(vis);
    num_times      = oskar_vis_block_num_times(vis);

    /* Get factor for conversion of sigma to SEFD. */
    sefd_conversion = sqrt(2.0*channel_bandwidth_hz * time_int_sec);

    /* If we are adding noise directly to Stokes I, the noise is defined
     * as single dipole noise, so we have to divide by sqrt(2) to take into
     * account of the two different dipoles that go into the calculation of
     * Stokes I. For polarised visibilities this is not required, as this
     * falls out naturally when evaluating Stokes I from the dipole
     * correlations (i.e. I = 0.5 (XX+YY) ). */

    switch (oskar_mem_type(oskar_vis_block_cross_correlations(vis)))
    {
    case OSKAR_SINGLE_COMPLEX:
    {
        const float* station_std;
        float2* data;
        station_std = oskar_mem_float_const(station_std_dev, status);
        for (t = 0; t < num_times; ++t)
        {
            if (have_crosscorr)
            {
                /* Cross-correlation noise. */
                block_start = num_baselines * (num_channels * t + channel_idx);
                data = (float2*) xcorr_ptr + block_start;
                for (a1 = 0, b = 0; a1 < num_stations; ++a1)
                {
                    for (a2 = a1 + 1; a2 < num_stations; ++b, ++a2)
                    {
                        oskar_random_gaussian2(seed, c++, block_idx, rnd);
                        std = sqrt(station_std[a1] * station_std[a2]) * inv_sqrt2;
                        data[b].x += std * rnd[0];
                        data[b].y += std * rnd[1];
                    }
                }
            }

            if (have_autocorr)
            {
                /* Autocorrelation noise. Phases are all zero after
                 * autocorrelation, so ignore the imaginary components. */
                block_start = num_stations * (num_channels * t + channel_idx);
                data = (float2*) acorr_ptr + block_start;
                for (a1 = 0; a1 < num_stations; ++a1)
                {
                    oskar_random_gaussian2(seed, c++, block_idx, rnd);
                    std = station_std[a1];
                    mean = sqrt(2.0)*station_std[a1];
                    data[a1].x += std * rnd[0] + mean * sefd_conversion;
                }
            }
        }
        break;
    }
    case OSKAR_SINGLE_COMPLEX_MATRIX:
    {
        const float* station_std;
        float4c* data;
        station_std = oskar_mem_float_const(station_std_dev, status);
        for (t = 0; t < num_times; ++t)
        {
            if (have_crosscorr)
            {
                /* Cross-correlation noise. */
                block_start = num_baselines * (num_channels * t + channel_idx);
                data = (float4c*) xcorr_ptr + block_start;
                for (a1 = 0, b = 0; a1 < num_stations; ++a1)
                {
                    for (a2 = a1 + 1; a2 < num_stations; ++b, ++a2)
                    {
                        oskar_random_gaussian4(seed, c++, block_idx, 0, 0, rnd);
                        oskar_random_gaussian4(seed, c++, block_idx, 0, 0, rnd + 4);
                        std = sqrt(station_std[a1] * station_std[a2]);
                        data[b].a.x += std * rnd[0];
                        data[b].a.y += std * rnd[1];
                        data[b].b.x += std * rnd[2];
                        data[b].b.y += std * rnd[3];
                        data[b].c.x += std * rnd[4];
                        data[b].c.y += std * rnd[5];
                        data[b].d.x += std * rnd[6];
                        data[b].d.y += std * rnd[7];
                    }
                }
            }

            if (have_autocorr)
            {
                /* Autocorrelation noise. Phases are all zero after
                 * autocorrelation, so ignore the imaginary components. */
                block_start = num_stations * (num_channels * t + channel_idx);
                data = (float4c*) acorr_ptr + block_start;
                for (a1 = 0; a1 < num_stations; ++a1)
                {
                    oskar_random_gaussian4(seed, c++, block_idx, 0, 0, rnd);
                    oskar_random_gaussian4(seed, c++, block_idx, 0, 0, rnd + 4);
                    std = station_std[a1] * sqrt(2.0);
                    mean = std * sefd_conversion;
                    data[a1].a.x += std * rnd[0] + mean;
                    data[a1].b.x += std * rnd[1];
                    data[a1].b.y += std * rnd[2];
                    data[a1].c.x += std * rnd[3];
                    data[a1].c.y += std * rnd[4];
                    data[a1].d.x += std * rnd[5] + mean;
                }
            }
        }
        break;
    }
    case OSKAR_DOUBLE_COMPLEX:
    {
        const double* station_std;
        double2* data;
        station_std = oskar_mem_double_const(station_std_dev, status);
        for (t = 0; t < num_times; ++t)
        {
            if (have_crosscorr)
            {
                /* Cross-correlation noise. */
                block_start = num_baselines * (num_channels * t + channel_idx);
                data = (double2*) xcorr_ptr + block_start;
                for (a1 = 0, b = 0; a1 < num_stations; ++a1)
                {
                    for (a2 = a1 + 1; a2 < num_stations; ++b, ++a2)
                    {
                        oskar_random_gaussian2(seed, c++, block_idx, rnd);
                        std = sqrt(station_std[a1] * station_std[a2]) * inv_sqrt2;
                        data[b].x += std * rnd[0];
                        data[b].y += std * rnd[1];
                    }
                }
            }

            if (have_autocorr)
            {
                /* Autocorrelation noise. Phases are all zero after
                 * autocorrelation, so ignore the imaginary components. */
                block_start = num_stations * (num_channels * t + channel_idx);
                data = (double2*) acorr_ptr + block_start;
                for (a1 = 0; a1 < num_stations; ++a1)
                {
                    oskar_random_gaussian2(seed, c++, block_idx, rnd);
                    std  = station_std[a1];
                    mean = station_std[a1] * sefd_conversion * sqrt(2.0);
                    data[a1].x += std * rnd[0] + mean;
                }
            }
        }
        break;
    }
    case OSKAR_DOUBLE_COMPLEX_MATRIX:
    {
        const double* station_std;
        double4c* data;
        station_std = oskar_mem_double_const(station_std_dev, status);
        for (t = 0; t < num_times; ++t)
        {
            if (have_crosscorr)
            {
                /* Cross-correlation noise. */
                block_start = num_baselines * (num_channels * t + channel_idx);
                data = (double4c*) xcorr_ptr + block_start;
                for (a1 = 0, b = 0; a1 < num_stations; ++a1)
                {
                    for (a2 = a1 + 1; a2 < num_stations; ++b, ++a2)
                    {
                        oskar_random_gaussian4(seed, c++, block_idx, 0, 0, rnd);
                        oskar_random_gaussian4(seed, c++, block_idx, 0, 0, rnd + 4);
                        std = sqrt(station_std[a1] * station_std[a2]);
                        data[b].a.x += std * rnd[0];
                        data[b].a.y += std * rnd[1];
                        data[b].b.x += std * rnd[2];
                        data[b].b.y += std * rnd[3];
                        data[b].c.x += std * rnd[4];
                        data[b].c.y += std * rnd[5];
                        data[b].d.x += std * rnd[6];
                        data[b].d.y += std * rnd[7];
                    }
                }
            }

            if (have_autocorr)
            {
                /* Autocorrelation noise. Phases are all zero after
                 * autocorrelation, so ignore the imaginary components. */
                block_start = num_stations * (num_channels * t + channel_idx);
                data = (double4c*) acorr_ptr + block_start;
                for (a1 = 0; a1 < num_stations; ++a1)
                {
                    oskar_random_gaussian4(seed, c++, block_idx, 0, 0, rnd);
                    oskar_random_gaussian4(seed, c++, block_idx, 0, 0, rnd+4);
                    std  = station_std[a1]*sqrt(2.0);
                    mean = std * sefd_conversion;
                    data[a1].a.x += std * rnd[0] + mean;
                    data[a1].b.x += std * rnd[1];
                    data[a1].b.y += std * rnd[2];
                    data[a1].c.x += std * rnd[3];
                    data[a1].c.y += std * rnd[4];
                    data[a1].d.x += std * rnd[5] + mean;
                }
            }
        }
        break;
    }
    };
}

void oskar_vis_block_add_system_noise(oskar_VisBlock* vis,
        const oskar_VisHeader* header, const oskar_Telescope* telescope,
        unsigned int block_index, oskar_Mem* station_work, int* status)
{
    int c, num_channels;
    unsigned int seed;
    double freq_hz, freq_start_hz, freq_inc_hz;
    double channel_bandwidth_hz, time_int_sec;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Return if noise is not enabled. */
    if (!oskar_telescope_noise_enabled(telescope)) return;

    /* Check baseline dimensions match. */
    if (oskar_telescope_num_baselines(telescope) !=
            oskar_vis_block_num_baselines(vis))
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Get frequency start and increment. */
    seed                 = oskar_telescope_noise_seed(telescope);
    num_channels         = oskar_vis_block_num_channels(vis);
    channel_bandwidth_hz = oskar_vis_header_channel_bandwidth_hz(header);
    time_int_sec         = oskar_vis_header_time_average_sec(header);
    freq_start_hz        = oskar_vis_header_freq_start_hz(header);
    freq_inc_hz          = oskar_vis_header_freq_inc_hz(header);

    /* Apply noise to each channel. */
    for (c = 0; c < num_channels; ++c)
    {
        freq_hz = freq_start_hz + c * freq_inc_hz;
        oskar_get_station_std_dev_for_channel(station_work, freq_hz,
                telescope, status);
        oskar_vis_block_apply_noise(vis, station_work, seed,
                block_index, c, channel_bandwidth_hz, time_int_sec, status);
    }
}

#ifdef __cplusplus
}
#endif
