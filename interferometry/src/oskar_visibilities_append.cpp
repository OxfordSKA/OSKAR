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
#include "interferometry/oskar_visibilities_append.h"
#include "utility/oskar_Mem.h"

int oskar_visibilties_append(oskar_Visibilities* dst, const oskar_Visibilities* src)
{
    int error = 0;

    int num_samples = src->num_baselines * src->num_times * src->num_channels;
    if (num_samples > src->baseline_u.n_elements() ||
            num_samples > src->baseline_v.n_elements() ||
            num_samples > src->baseline_w.n_elements() ||
            num_samples > src->amplitude.n_elements())
    {
        return OSKAR_ERR_DIMENSION_MISMATCH;
    }

    if (dst->num_baselines != src->num_baselines)
    {
        return OSKAR_ERR_DIMENSION_MISMATCH;
    }

    // NOTE: Appending visibilities for different numbers of channels in each
    // structure is currently not allowed as it would require extra information
    // on the channel id's being appended and some memory reorder.
    if (dst->num_channels != src->num_channels)
    {
        return OSKAR_ERR_DIMENSION_MISMATCH;
    }

    int location = src->location();
    error = dst->baseline_u.append(src->baseline_u.data, src->baseline_u.type(),
            location, num_samples);
    if (error) return error;
    error = dst->baseline_v.append(src->baseline_v.data, src->baseline_v.type(),
            location, num_samples);
    if (error) return error;
    error = dst->baseline_w.append(src->baseline_w.data, src->baseline_w.type(),
            location, num_samples);
    if (error) return error;
    error = dst->amplitude.append(src->amplitude.data, src->amplitude.type(),
            location, num_samples);
    if (error) return error;

    dst->num_times += src->num_times;

    return error;
}



