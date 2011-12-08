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

#include "math/oskar_spline_data_evaluate.h"
#include "math/oskar_spline_surface_evaluate.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_spline_data_evaluate(oskar_Mem* output,
		const oskar_SplineData* spline, const oskar_Mem* x, const oskar_Mem* y)
{
	int err = 0, nx, ny, kx, ky, num_points, type, location;

	/* Check arrays are consistent. */
	num_points = output->private_num_elements;
	if (num_points != x->private_num_elements ||
			num_points != y->private_num_elements)
		return OSKAR_ERR_DIMENSION_MISMATCH;

	/* Check type. */
	type = output->private_type;
	if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
		return OSKAR_ERR_BAD_DATA_TYPE;
	if (type != x->private_type || type != y->private_type)
		return OSKAR_ERR_TYPE_MISMATCH;

	/* Check location. */
	location = output->private_location;
	if (location != OSKAR_LOCATION_CPU)
		return OSKAR_ERR_BAD_LOCATION;
	if (location != spline->coeff.private_location ||
			location != spline->knots_x.private_location ||
			location != spline->knots_y.private_location ||
			location != x->private_location ||
			location != y->private_location)
		return OSKAR_ERR_BAD_LOCATION;

	/* Get common data. */
	nx = spline->num_knots_x;
	ny = spline->num_knots_y;
	kx = spline->degree_x;
	ky = spline->degree_y;

	if (type == OSKAR_SINGLE)
	{
		const float *tx, *ty, *c, *px, *py;
		float* z;
		tx = (const float*)spline->knots_x.data;
		ty = (const float*)spline->knots_y.data;
		c = (const float*)spline->coeff.data;
		px = (const float*)x->data;
		py = (const float*)y->data;
		z = (float*)output->data;

		err = oskar_spline_surface_evaluate_f(tx, nx, ty, ny, c, kx, ky,
				num_points, px, py, z);
	}
	else if (type == OSKAR_DOUBLE)
	{
		return OSKAR_ERR_UNKNOWN;
	}

    return err;
}

#ifdef __cplusplus
}
#endif
