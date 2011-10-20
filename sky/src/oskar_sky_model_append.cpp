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
int oskar_sky_model_append(oskar_SkyModel* to, const oskar_SkyModel* from)
{
    int error = 0;

    if (from->num_sources > from->RA.n_elements()  ||
        from->num_sources > from->Dec.n_elements() ||
        from->num_sources > from->I.n_elements() ||
        from->num_sources > from->Q.n_elements() ||
        from->num_sources > from->U.n_elements() ||
        from->num_sources > from->V.n_elements() ||
        from->num_sources > from->reference_freq.n_elements() ||
        from->num_sources > from->spectral_index.n_elements())
    {
        return -1;
    }

    // Append to the sky model.
    int location = from->location();
    int num_sources = from->num_sources;
    error = to->RA.append(from->RA.data, location, num_sources);
    if (error) return error;
    error = to->Dec.append(from->Dec.data, location, num_sources);
    if (error) return error;
    error = to->I.append(from->I.data, location, num_sources);
    if (error) return error;
    error = to->Q.append(from->Q.data, location, num_sources);
    if (error) return error;
    error = to->U.append(from->U.data, location, num_sources);
    if (error) return error;
    error = to->V.append(from->V.data, location, num_sources);
    if (error) return error;
    error = to->reference_freq.append(from->reference_freq.data, location, num_sources);
    if (error) return error;
    error = to->spectral_index.append(from->spectral_index.data, location, num_sources);
    if (error) return error;

    // Update the number of sources
    to->num_sources += from->num_sources;

    // Resize work arrays
    error = to->rel_l.resize(num_sources);
    if (error) return error;
    error = to->rel_m.resize(num_sources);
    if (error) return error;
    error = to->rel_n.resize(num_sources);
    if (error) return error;
    error = to->hor_l.resize(num_sources);
    if (error) return error;
    error = to->hor_m.resize(num_sources);
    if (error) return error;
    error = to->hor_n.resize(num_sources);
    if (error) return error;

    return error;
}

