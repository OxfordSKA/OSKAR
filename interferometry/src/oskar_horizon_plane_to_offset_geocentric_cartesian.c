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

#include "interferometry/oskar_horizon_plane_to_offset_geocentric_cartesian.h"

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// Double precision.
void oskar_horizon_plane_to_offset_geocentric_cartesian_d(int n,
        const double* x_horizon, const double* y_horizon,
        const double* z_horizon, double longitude, double latitude,
        double* x, double* y, double* z)
{
	// Precompute some trig.
	double sin_l = sin(longitude);
	double cos_l = cos(longitude);
	double sin_p = sin(latitude);
	double cos_p = cos(latitude);

	// Loop over points.
	int i = 0;
	double X, Y, Z, xi, yi, zi;
	for (i = 0; i < n; ++i)
	{
		xi = x_horizon[i];
		yi = y_horizon[i];
		zi = z_horizon[i];
		X = -xi * sin_l - yi * sin_p * cos_l + zi * cos_p * cos_l;
		Y =  xi * cos_l - yi * sin_p * sin_l + zi * cos_p * sin_l;
		Z =  yi * cos_p + zi * sin_p;
		x[i] = X;
		y[i] = Y;
		z[i] = Z;
	}
}

// Single precision.
void oskar_horizon_plane_to_offset_geocentric_cartesian_f(int n,
        const float* x_horizon, const float* y_horizon,
        const float* z_horizon, float longitude, float latitude,
        float* x, float* y, float* z)
{
	// Precompute some trig.
	float sin_l = sinf(longitude);
	float cos_l = cosf(longitude);
	float sin_p = sinf(latitude);
	float cos_p = cosf(latitude);

	// Loop over points.
	int i = 0;
	float X, Y, Z, xi, yi, zi;
	for (i = 0; i < n; ++i)
	{
		xi = x_horizon[i];
		yi = y_horizon[i];
		zi = z_horizon[i];
		X = -xi * sin_l - yi * sin_p * cos_l + zi * cos_p * cos_l;
		Y =  xi * cos_l - yi * sin_p * sin_l + zi * cos_p * sin_l;
		Z =  yi * cos_p + zi * sin_p;
		x[i] = X;
		y[i] = Y;
		z[i] = Z;
	}
}

#ifdef __cplusplus
}
#endif
