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

#include "sky/cudak/oskar_cudak_stokes_to_local_field_vectors.h"

// Single precision.

__global__
void oskar_cudak_stokes_to_local_field_vectors_f(int ns, const float* ra,
        const float* dec, float cosLat, float sinLat, float lst,
        float* l, float* m, float* n)
{
	// Stokes to linear components.
	float s, c, t;
	t = 0.5f * atan2f(s_U, s_Q);
	sincosf(t, &s, &c);
	t = hypotf(s_Q, s_U);
	float e_alpha = t * s;
	float e_delta = t * c;

	// Get phi and theta angles from horizontal direction cosines.
	float phi = atan2(s_l, s_m);
	t = hypotf(s_l, s_m);
	float theta = atan2(t, s_n);

	t = 0.5f * (s_I - p1);
}

// Double precision.

__global__
void oskar_cudak_stokes_to_local_field_vectors_d(int ns, const double* ra,
        const double* dec, double cosLat, double sinLat, double lst,
        double* l, double* m, double* n)
{
}
