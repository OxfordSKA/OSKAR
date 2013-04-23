/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#include "interferometry/oskar_visibilities_add_system_noise.h"
#include "math/oskar_random_gaussian.h"
#include "math/oskar_find_closest_match.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_vector_types.h"

#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define DAYS_2_SEC 86400.0

#ifdef __cplusplus
extern "C" {
#endif

void oskar_visibilities_add_system_noise(oskar_Visibilities* vis,
        const oskar_TelescopeModel* telescope, unsigned seed, int* status)
{
    oskar_Mem ant1, ant2;
    int location = OSKAR_LOCATION_CPU;
    int a1, a2, c, t, b, is1, is2, idx;
    double r1, r2, s1, s2, std, vis_freq, mean = 0.0;
    oskar_Mem *noise_freq, *noise_rms;

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
    oskar_mem_init(&ant1, OSKAR_INT, location, vis->num_baselines, 1, status);
    oskar_mem_init(&ant2, OSKAR_INT, location, vis->num_baselines, 1, status);
    if (*status) return;
    for (b = 0, a1 = 0; a1 < telescope->num_stations; ++a1)
    {
        for (a2 = (a1 + 1); a2 < telescope->num_stations; ++a2)
        {
            ((int*)ant1.data)[b] = a1;
            ((int*)ant2.data)[b] = a2;
            b++;
        }
    }

    for (idx = 0, c = 0; c < vis->num_channels; ++c)
    {
        vis_freq = vis->freq_start_hz + c * vis->freq_inc_hz;

        for (t = 0; t < vis->num_times; ++t)
        {
            for (b = 0; b < vis->num_baselines; ++b)
            {
                /* Retrieve the std.dev. for the baseline antennas. */
                a1 = ((int*)ant1.data)[b];
                noise_freq = &telescope->station[a1].noise.frequency;
                noise_rms = &telescope->station[a1].noise.rms;
                oskar_find_closest_match(&is1, vis_freq, noise_freq, status);
                if (noise_freq->type == OSKAR_DOUBLE)
                    s1 = ((double*)noise_rms->data)[is1];
                else
                    s1 = ((float*)noise_rms->data)[is1];

                a2 = ((int*)ant2.data)[b];
                noise_freq = &telescope->station[a2].noise.frequency;
                noise_rms = &telescope->station[a2].noise.rms;
                oskar_find_closest_match(&is2, vis_freq, noise_freq, status);
                if (noise_freq->type == OSKAR_DOUBLE)
                    s2 = ((double*)noise_rms->data)[is2];
                else
                    s2 = ((float*)noise_rms->data)[is2];

                /* Combine antenna std.devs. to evaluate the baseline std.dev.
                 * See Wrobel & Walker (1999) */
                std = sqrt(s1*s2);

                /* Apply noise */
                switch (vis->amplitude.type)
                {
                    case OSKAR_SINGLE_COMPLEX:
                    {
                        float2* amps_ = (float2*)vis->amplitude.data;
                        r1 = oskar_random_gaussian(&r2);
                        amps_[idx].x += r1 * std + mean;
                        amps_[idx].y += r2 * std + mean;
                        break;
                    }
                    case OSKAR_SINGLE_COMPLEX_MATRIX:
                    {
                        float4c* amps_ = (float4c*)vis->amplitude.data;
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
                    case OSKAR_DOUBLE_COMPLEX:
                    {
                        double2* amps_ = (double2*)vis->amplitude.data;
                        r1 = oskar_random_gaussian(&r2);
                        amps_[idx].x += r1 * std + mean;
                        amps_[idx].y += r2 * std + mean;
                        break;
                    }
                    case OSKAR_DOUBLE_COMPLEX_MATRIX:
                    {
                        double4c* amps_ = (double4c*)vis->amplitude.data;
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

    oskar_mem_free(&ant1, status);
    oskar_mem_free(&ant2, status);
}

#ifdef __cplusplus
}
#endif
