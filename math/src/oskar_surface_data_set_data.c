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

#include "math/oskar_surface_data_set_data.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_surface_data_set_data(oskar_SurfaceData* data,
		int i, double real, double imag)
{
	/* Check index is in range. */
	if (i >= data->re.private_num_elements ||
			i >= data->im.private_num_elements)
		return OSKAR_ERR_OUT_OF_RANGE;

	/* Store the data. */
	if (data->re.private_type == OSKAR_SINGLE &&
			data->im.private_type == OSKAR_SINGLE)
	{
		((float*)data->re.data)[i] = real;
		((float*)data->im.data)[i] = imag;
	}
	else if (data->re.private_type == OSKAR_DOUBLE &&
			data->im.private_type == OSKAR_DOUBLE)
	{
		((double*)data->re.data)[i] = real;
		((double*)data->im.data)[i] = imag;
	}
	else
		return OSKAR_ERR_BAD_DATA_TYPE;

    return 0;
}

#ifdef __cplusplus
}
#endif
