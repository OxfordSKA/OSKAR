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

#include "sky/oskar_sky_model_compute_relative_lmn.h"
#include "sky/oskar_cuda_ra_dec_to_relative_lmn.h"
#include <cstdlib>

#ifdef __cplusplus
extern "C"
#endif
int oskar_sky_model_compute_relative_lmn(oskar_SkyModel* sky, double ra0,
		double dec0)
{
    // Check for sane inputs.
    if (sky == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    // Check that data is on the GPU.
    if (sky->RA.location() != OSKAR_LOCATION_GPU ||
    		sky->Dec.location() != OSKAR_LOCATION_GPU ||
    		sky->rel_l.location() != OSKAR_LOCATION_GPU ||
    		sky->rel_m.location() != OSKAR_LOCATION_GPU ||
    		sky->rel_n.location() != OSKAR_LOCATION_GPU)
        return OSKAR_ERR_BAD_LOCATION;

    // Check the types are OK.
    int error = 0;
    if (sky->RA.type() == OSKAR_SINGLE &&
    		sky->Dec.type() == OSKAR_SINGLE &&
    		sky->rel_l.type() == OSKAR_SINGLE &&
    		sky->rel_m.type() == OSKAR_SINGLE &&
    		sky->rel_n.type() == OSKAR_SINGLE)
    {
		// Convert the coordinates (single precision).
    	error = oskar_cuda_ra_dec_to_relative_lmn_f(sky->num_sources, sky->RA,
				sky->Dec, (float)ra0, (float)dec0, sky->rel_l, sky->rel_m,
				sky->rel_n);
    }
    else if (sky->RA.type() == OSKAR_DOUBLE &&
    		sky->Dec.type() == OSKAR_DOUBLE &&
    		sky->rel_l.type() == OSKAR_DOUBLE &&
    		sky->rel_m.type() == OSKAR_DOUBLE &&
    		sky->rel_n.type() == OSKAR_DOUBLE)
    {
		// Convert the coordinates (double precision).
    	error = oskar_cuda_ra_dec_to_relative_lmn_d(sky->num_sources, sky->RA,
				sky->Dec, ra0, dec0, sky->rel_l, sky->rel_m, sky->rel_n);
    }
    else
    {
    	return OSKAR_ERR_TYPE_MISMATCH;
    }

    return error;
}
