/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#include <private_vis.h>
#include <oskar_vis.h>

#include <oskar_random_gaussian.h>
#include <oskar_find_closest_match.h>

#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>

#define DAYS_2_SEC 86400.0

#ifdef __cplusplus
extern "C" {
#endif

void oskar_vis_add_system_noise(oskar_Vis* vis,
        const oskar_Telescope* telescope, unsigned seed, int* status)
{
    int *ant1, *ant2;
    int a1, a2, c, t, b, is1, is2, idx, num_baselines, num_stations;
    oskar_Mem *vis_amp;

    /* Check all inputs. */
    if (!vis || !telescope || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Seed the random number generator */
    srand(seed);

    /* Evaluate baseline antenna IDs. Needed for lookup of station std.dev. */
    num_baselines = vis->num_baselines;
    num_stations = oskar_telescope_num_stations(telescope);
    vis_amp = vis->amplitude;
    ant1 = malloc(num_baselines * sizeof(int));
    ant2 = malloc(num_baselines * sizeof(int));
    for (b = 0, a1 = 0; a1 < num_stations; ++a1)
    {
        for (a2 = (a1 + 1); a2 < num_stations; ++a2)
        {
            ant1[b] = a1;
            ant2[b] = a2;
            b++;
        }
    }

    for (idx = 0, c = 0; c < vis->num_channels; ++c)
    {
        double vis_freq, std, r1, r2, s1, s2, mean = 0.0;

        vis_freq = vis->freq_start_hz + c * vis->freq_inc_hz;

        for (t = 0; t < vis->num_times; ++t)
        {
            for (b = 0; b < num_baselines; ++b)
            {
                const oskar_Mem *noise_freq, *noise_rms;
                const oskar_Station *st;

                /* Retrieve the STD for the baseline antennas. */
                st = oskar_telescope_station_const(telescope, ant1[b]);
                noise_freq = oskar_station_noise_freq_hz_const(st);
                noise_rms = oskar_station_noise_rms_jy_const(st);
                is1 = oskar_find_closest_match(vis_freq, noise_freq, status);
                if (oskar_mem_type(noise_freq) == OSKAR_DOUBLE)
                    s1 = oskar_mem_double_const(noise_rms, status)[is1];
                else
                    s1 = oskar_mem_float_const(noise_rms, status)[is1];

                st = oskar_telescope_station_const(telescope, ant2[b]);
                noise_freq = oskar_station_noise_freq_hz_const(st);
                noise_rms = oskar_station_noise_rms_jy_const(st);
                is2 = oskar_find_closest_match(vis_freq, noise_freq, status);
                if (oskar_mem_type(noise_freq) == OSKAR_DOUBLE)
                    s2 = oskar_mem_double_const(noise_rms, status)[is2];
                else
                    s2 = oskar_mem_float_const(noise_rms, status)[is2];

                /* Combine antenna STD to evaluate the baseline STD
                 * See Wrobel & Walker (1999) */
                std = sqrt(s1*s2);

                /* Apply noise */
                switch (oskar_mem_type(vis_amp))
                {
                    case OSKAR_SINGLE_COMPLEX: /* Scalar amps. = Stokes-I mode */
                    {
                        float2* amps_;
                        amps_ = oskar_mem_float2(vis_amp, status);
                        r1 = oskar_random_gaussian(&r2);
                        /* As we are adding noise directly to stokes I
                         * and the noise is defined as single dipole noise
                         * we have to divide by sqrt(2) to take into the account
                         * of the two different dipoles that go into the
                         * calculation of stokes-I. Note that for polarised mode
                         * this is not required (npols == 4) as this falls out
                         * naturally when evaluating stokes-I from the
                         * dipole correlations (ie. I = 0.5 (XX+YY) ).
                         */
                        std = std/sqrt(2);
                        amps_[idx].x += r1 * std + mean;
                        amps_[idx].y += r2 * std + mean;
                        break;
                    }
                    case OSKAR_SINGLE_COMPLEX_MATRIX:
                    {
                        float4c* amps_;
                        amps_ = oskar_mem_float4c(vis_amp, status);
                        r1 = oskar_random_gaussian(&r2);
                        amps_[idx].a.x += r1 * std + mean;
                        amps_[idx].a.y += r2 * std + mean;
                        r1 = oskar_random_gaussian(&r2);
                        amps_[idx].b.x += r1 * std + mean;
                        amps_[idx].b.y += r2 * std + mean;
                        r1 = oskar_random_gaussian(&r2);
                        amps_[idx].c.x += r1 * std + mean;
                        amps_[idx].c.y += r2 * std + mean;
                        r1 = oskar_random_gaussian(&r2);
                        amps_[idx].d.x += r1 * std + mean;
                        amps_[idx].d.y += r2 * std + mean;
                        break;
                    }
                    case OSKAR_DOUBLE_COMPLEX: /* Scalar amps. = Stokes-I mode */
                    {
                        double2* amps_;
                        amps_ = oskar_mem_double2(vis_amp, status);
                        r1 = oskar_random_gaussian(&r2);
                        std = std/sqrt(2);
                        amps_[idx].x += r1 * std + mean;
                        amps_[idx].y += r2 * std + mean;
                        break;
                    }
                    case OSKAR_DOUBLE_COMPLEX_MATRIX:
                    {
                        double4c* amps_;
                        amps_ = oskar_mem_double4c(vis_amp, status);
                        r1 = oskar_random_gaussian(&r2);
                        amps_[idx].a.x += r1 * std + mean;
                        amps_[idx].a.y += r2 * std + mean;
                        r1 = oskar_random_gaussian(&r2);
                        amps_[idx].b.x += r1 * std + mean;
                        amps_[idx].b.y += r2 * std + mean;
                        r1 = oskar_random_gaussian(&r2);
                        amps_[idx].c.x += r1 * std + mean;
                        amps_[idx].c.y += r2 * std + mean;
                        r1 = oskar_random_gaussian(&r2);
                        amps_[idx].d.x += r1 * std + mean;
                        amps_[idx].d.y += r2 * std + mean;
                        break;
                    }
                    default:
                    {
                        *status = OSKAR_ERR_BAD_DATA_TYPE;
                        break;
                    }
                };
                ++idx;
            }
        }
    }

    free(ant1);
    free(ant2);
}

#ifdef __cplusplus
}
#endif
