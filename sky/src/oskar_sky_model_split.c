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


#include "sky/oskar_sky_model_split.h"
#include "math/oskar_round_robin.h"
#include "sky/oskar_sky_model_get_ptr.h"

#include "math.h"
#include "stdlib.h"
#include "stdio.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_sky_model_split(oskar_SkyModel** out, int* num_out, int max_sources_out,
        const oskar_SkyModel* in)
{
    int offset, size, i, error;

    if (in == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    *num_out = (int)ceil((double)in->num_sources/max_sources_out);
    *out = (oskar_SkyModel*)malloc(*num_out * sizeof(oskar_SkyModel));

    for (i = 0; i < *num_out; ++i)
    {
        oskar_round_robin(in->num_sources, *num_out, i, &size, &offset);
        error = oskar_sky_model_get_ptr(&(*out)[i], in, offset, size);
        if (error) return error;
    }

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
