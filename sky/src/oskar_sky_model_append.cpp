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
 *    and/or src materials provided with the distribution.
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
#include "sky/oskar_sky_model_append.h"
#include "utility/oskar_Mem.h"

#ifdef __cplusplus
extern "C"
#endif
int oskar_sky_model_append(oskar_SkyModel* dst, const oskar_SkyModel* src)
{
    int error = 0;

    int num_sources = src->num_sources;

    if (num_sources > src->RA.num_elements()  ||
        num_sources > src->Dec.num_elements() ||
        num_sources > src->I.num_elements() ||
        num_sources > src->Q.num_elements() ||
        num_sources > src->U.num_elements() ||
        num_sources > src->V.num_elements() ||
        num_sources > src->reference_freq.num_elements() ||
        num_sources > src->spectral_index.num_elements())
    {
        return OSKAR_ERR_DIMENSION_MISMATCH;
    }

    // Append to the sky model.
    int location = src->location();
    int type = src->type();
    error = dst->RA.append(src->RA.data, type, location, num_sources);
    if (error) return error;
    error = dst->Dec.append(src->Dec.data, type, location, num_sources);
    if (error) return error;
    error = dst->I.append(src->I.data, type, location, num_sources);
    if (error) return error;
    error = dst->Q.append(src->Q.data, type, location, num_sources);
    if (error) return error;
    error = dst->U.append(src->U.data, type, location, num_sources);
    if (error) return error;
    error = dst->V.append(src->V.data, type, location, num_sources);
    if (error) return error;
    error = dst->reference_freq.append(src->reference_freq.data, type, location, num_sources);
    if (error) return error;
    error = dst->spectral_index.append(src->spectral_index.data, type, location, num_sources);
    if (error) return error;

    // Update the number of sources
    dst->num_sources += src->num_sources;

    error = dst->rel_l.resize(num_sources);
    if (error) return error;
    error = dst->rel_m.resize(num_sources);
    if (error) return error;
    error = dst->rel_n.resize(num_sources);
    if (error) return error;

    return error;
}

