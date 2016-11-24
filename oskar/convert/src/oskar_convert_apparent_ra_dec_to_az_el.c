/*
 * Copyright (c) 2013-2014, The University of Oxford
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

#include "convert/oskar_convert_apparent_ra_dec_to_az_el.h"
#include "convert/oskar_convert_apparent_ra_dec_to_enu_directions.h"
#include "convert/oskar_convert_enu_directions_to_az_el.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif


/* Single precision. */
void oskar_convert_apparent_ra_dec_to_az_el_f(int n, const float* ra,
        const float* dec, float lst, float lat, float* work, float* az,
        float* el)
{
    oskar_convert_apparent_ra_dec_to_enu_directions_f(n, ra, dec, lst,
            lat, az, el, work);
    oskar_convert_enu_directions_to_az_el_f(n, az, el, work, az, el);
}


/* Double precision. */
void oskar_convert_apparent_ra_dec_to_az_el_d(int n, const double* ra,
        const double* dec, double lst, double lat, double* work, double* az,
        double* el)
{
    oskar_convert_apparent_ra_dec_to_enu_directions_d(n, ra, dec, lst,
            lat, az, el, work);
    oskar_convert_enu_directions_to_az_el_d(n, az, el, work, az, el);
}


#ifdef __cplusplus
}
#endif
