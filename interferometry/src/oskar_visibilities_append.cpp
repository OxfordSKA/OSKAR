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

    int num_samples = src->num_channels * src->num_times * src->num_baselines;
    if (num_samples > src->uu_metres.num_elements() ||
            num_samples > src->vv_metres.num_elements() ||
            num_samples > src->ww_metres.num_elements() ||
            num_samples > src->amplitude.num_elements())
    {
        return OSKAR_ERR_DIMENSION_MISMATCH;
    }

    // For the case of appending to a completely empty visibility structure
    // initialise its dimensions to the size of the data being appended.
    if (dst->num_channels == 0 && dst->num_times == 0 && dst->num_baselines == 0)
    {
        dst->num_channels  = src->num_channels;
        dst->num_baselines = src->num_baselines;
    }


    if (dst->num_baselines != src->num_baselines)
    {
        return OSKAR_ERR_DIMENSION_MISMATCH;
    }

    error = dst->uu_metres.append(src->uu_metres.data, src->uu_metres.type(),
            src->uu_metres.location(), num_samples);
    if (error) return error;
    error = dst->vv_metres.append(src->vv_metres.data, src->vv_metres.type(),
            src->vv_metres.location(), num_samples);
    if (error) return error;
    error = dst->ww_metres.append(src->ww_metres.data, src->ww_metres.type(),
            src->ww_metres.location(), num_samples);
    if (error) return error;
    error = dst->amplitude.append(src->amplitude.data, src->amplitude.type(),
            src->amplitude.location(), num_samples);
    if (error) return error;

    dst->num_times += src->num_times;

    return error;
}



