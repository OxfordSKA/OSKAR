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

#include "interferometry/oskar_telescope_model_location.h"
#include "interferometry/oskar_TelescopeModel.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_telescope_model_is_type(const oskar_TelescopeModel* telescope,
		int type)
{
    return (telescope->station_x.private_type == type &&
    		telescope->station_y.private_type == type &&
    		telescope->station_z.private_type == type);
}

int oskar_telescope_model_type(const oskar_TelescopeModel* telescope)
{
    if (telescope == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    if (oskar_telescope_model_is_type(telescope, OSKAR_DOUBLE))
        return OSKAR_DOUBLE;
    else if (oskar_telescope_model_is_type(telescope, OSKAR_SINGLE))
        return OSKAR_SINGLE;
    else
        return OSKAR_ERR_BAD_DATA_TYPE;
}

#ifdef __cplusplus
}
#endif
