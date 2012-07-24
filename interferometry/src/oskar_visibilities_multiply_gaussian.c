/*
 * Copyright (c) 2012, The University of Oxford
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

#include "oskar_visibilities_multiply_gaussian.h"
#include <math.h>

#ifndef M_2_SQRT_2_LN2
#define M_2_SQRT_2_LN2 2.35482004503094938202314 /* 2 sqrt(2 * log_e(2)) */
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846 /* pi */
#endif


#ifdef __cplusplus
extern "C" {
#endif

/* FIXME Is this function still used by anything? Can it be removed? */

/*
 *  http://www.aips.nrao.edu/cgi-bin/ZXHLP2.PL?UVMOD
 *  -- definition of rotation angle, and use of fwhm rather than stddev
 */
int oskar_visibilities_multiply_gaussian(oskar_Visibilities* vis,
        double fwhm_major_rad, double fwhm_minor_rad, double rotation_angle_rad)
{
    double sig_maj, sig_min, sig_maj_2, sig_min_2, a, b, c, cos_ra, sin_ra, sin_2ra;
    int i, num_vis;

    sig_maj = M_2_SQRT_2_LN2 / ( 2 * M_PI * fwhm_major_rad);
    sig_min = M_2_SQRT_2_LN2 / ( 2 * M_PI * fwhm_minor_rad);
        sig_min_2 = sig_min * sig_min;
    sig_maj_2 = sig_maj * sig_maj;
    cos_ra = cos(rotation_angle_rad);
    sin_ra = sin(rotation_angle_rad);
    sin_2ra = sin(2.0 * rotation_angle_rad);

    a = (cos_ra * cos_ra) / (2.0 * sig_min_2) + (sin_ra * sin_ra) / (2.0 * sig_maj_2);
    b = -sin_2ra / (4.0 * sig_min_2) + sin_2ra / (4.0 * sig_maj_2);
    c = (sin_ra * sin_ra) / (2.0 * sig_min_2) + (cos_ra * cos_ra) / (2.0 * sig_maj_2);

    printf("== gaussian param: a = %f, b = %f, c = %d\n", a, b, c);


    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
