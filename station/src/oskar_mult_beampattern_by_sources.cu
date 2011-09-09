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

#include "station/oskar_mult_beampattern_by_sources.h"
#include "math/cudak/oskar_cudak_vec_mul_cr.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mult_beampattern_by_source_field_amp_d(const unsigned num_stations,
        const oskar_SkyModelLocal_d* hd_sky, double2 * d_e_jones)
{
    int num_sources = hd_sky->num_sources;
    int num_threads = 256;
    int num_blocks  = (int)ceil((double) num_sources / num_threads);

    for (unsigned i = 0; i < num_stations; ++i)
    {
        double2* d_e_jones_station = d_e_jones + i * num_sources;
        oskar_cudak_vec_mul_cr_d <<< num_blocks, num_threads >>>
                (num_sources, d_e_jones_station, hd_sky->I, d_e_jones_station);
    }
}


void oskar_mult_beampattern_by_source_field_amp_f(const unsigned num_stations,
        const oskar_SkyModelLocal_f* hd_sky, float2* d_e_jones)
{
    int num_sources = hd_sky->num_sources;
    int num_threads = 256;
    int num_blocks  = (int)ceilf((float) num_sources / num_threads);

    for (unsigned i = 0; i < num_stations; ++i)
    {
        float2* d_e_jones_station = d_e_jones + i * num_sources;
        oskar_cudak_vec_mul_cr_f <<< num_blocks, num_threads >>>
                (num_sources, d_e_jones_station, hd_sky->I, d_e_jones_station);
    }
}

#ifdef __cplusplus
}
#endif
