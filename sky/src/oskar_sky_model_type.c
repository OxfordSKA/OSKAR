/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#include "sky/oskar_sky_model_type.h"
#include "sky/oskar_SkyModel.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_sky_model_is_type(const oskar_SkyModel* sky, int type)
{
    return (sky->RA.type == type &&
            sky->Dec.type == type &&
            sky->I.type == type &&
            sky->Q.type == type &&
            sky->U.type == type &&
            sky->V.type == type &&
            sky->reference_freq.type == type &&
            sky->spectral_index.type == type &&
            sky->l.type == type &&
            sky->m.type == type &&
            sky->n.type == type &&
            sky->FWHM_major.type == type &&
            sky->FWHM_minor.type == type &&
            sky->position_angle.type == type &&
            sky->gaussian_a.type == type &&
            sky->gaussian_b.type == type &&
            sky->gaussian_c.type == type);
}

int oskar_sky_model_type(const oskar_SkyModel* sky)
{
    if (sky == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    if (oskar_sky_model_is_type(sky, OSKAR_DOUBLE))
        return OSKAR_DOUBLE;
    else if (oskar_sky_model_is_type(sky, OSKAR_SINGLE))
        return OSKAR_SINGLE;
    else
        return OSKAR_ERR_BAD_DATA_TYPE;
}

#ifdef __cplusplus
}
#endif
