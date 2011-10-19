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
#include "sky/oskar_sky_model_resize.h"
#include "utility/oskar_Mem.h"

#ifdef __cplusplus
extern "C"
#endif
int oskar_sky_model_resize(oskar_SkyModel* sky, int num_sources)
{
    int error = 0;

    sky->num_sources = num_sources;

    // Resize the model data.
    error = sky->RA.resize(num_sources);
    if (error) return error;
    error = sky->Dec.resize(num_sources);
    if (error) return error;
    error = sky->I.resize(num_sources);
    if (error) return error;
    error = sky->Q.resize(num_sources);
    if (error) return error;
    error = sky->U.resize(num_sources);
    if (error) return error;
    error = sky->V.resize(num_sources);
    if (error) return error;
    error = sky->spectral_index.resize(num_sources);
    if (error) return error;
    error = sky->reference_freq.resize(num_sources);
    if (error) return error;

    // Resize the work buffers.
    error = sky->rel_l.resize(num_sources);
    if (error) return error;
    error = sky->rel_m.resize(num_sources);
    if (error) return error;
    error = sky->rel_n.resize(num_sources);
    if (error) return error;
    error = sky->hor_l.resize(num_sources);
    if (error) return error;
    error = sky->hor_m.resize(num_sources);
    if (error) return error;
    error = sky->hor_n.resize(num_sources);
    if (error) return error;

    return error;
}

