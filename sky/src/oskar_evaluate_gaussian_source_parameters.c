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


#include "sky/oskar_evaluate_gaussian_source_parameters.h"
#include "math/oskar_sph_to_lm.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define M_PI_2_2_LN_2 7.11941466249375271693034 /* pi^2 / (2 log_e(2)) */
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))


#ifdef __cplusplus
extern "C" {
#endif

int oskar_evaluate_gaussian_source_parameters(int num_sources,
        oskar_Mem* gaussian_a, oskar_Mem* gaussian_b, oskar_Mem* gaussian_c,
        oskar_Mem* FWHM_major, oskar_Mem* FWHM_minor, oskar_Mem* position_angle,
        oskar_Mem* RA, oskar_Mem* Dec, double ra0, double dec0)
{
    int i;
    double a, b, c;
    int type;
    double fwhm_maj, fwhm_min, pa;
    double cos_pa_2, sin_pa_2, sin_2pa;
    double inv_std_min_2, inv_std_maj_2;
    double ra, dec, delta_ra_maj, delta_dec_maj, delta_ra_min, delta_dec_min;
    oskar_Mem lon, lat, l, m;
    double cos_pa, sin_pa;
    double delta_l_maj, delta_l_min, delta_m_maj, delta_m_min, pa_lm;

    if (gaussian_a->type == OSKAR_DOUBLE &&
            gaussian_b->type == OSKAR_DOUBLE &&
            gaussian_c->type == OSKAR_DOUBLE &&
            FWHM_major->type == OSKAR_DOUBLE &&
            FWHM_minor->type == OSKAR_DOUBLE &&
            position_angle->type == OSKAR_DOUBLE &&
            RA->type == OSKAR_DOUBLE &&
            Dec->type == OSKAR_DOUBLE)
    {
        type = OSKAR_DOUBLE;
    }
    else if (gaussian_a->type == OSKAR_SINGLE &&
            gaussian_b->type == OSKAR_SINGLE &&
            gaussian_c->type == OSKAR_SINGLE &&
            FWHM_major->type == OSKAR_SINGLE &&
            FWHM_minor->type == OSKAR_SINGLE &&
            position_angle->type == OSKAR_SINGLE &&
            RA->type == OSKAR_SINGLE &&
            Dec->type == OSKAR_SINGLE)
    {
        type = OSKAR_SINGLE;
    }
    else
    {
        return OSKAR_ERR_BAD_DATA_TYPE;
    }

    oskar_mem_init(&lon, type, OSKAR_LOCATION_CPU, 4);
    oskar_mem_init(&lat, type, OSKAR_LOCATION_CPU, 4);
    oskar_mem_init(&l, type, OSKAR_LOCATION_CPU, 4);
    oskar_mem_init(&m, type, OSKAR_LOCATION_CPU, 4);

    if (gaussian_a == NULL || gaussian_b == NULL || gaussian_c == NULL ||
            FWHM_major == NULL || FWHM_minor == NULL || position_angle == NULL)
    {
        return OSKAR_ERR_INVALID_ARGUMENT;
    }

    if (num_sources > FWHM_major->num_elements ||
            num_sources > FWHM_minor->num_elements ||
            num_sources > position_angle->num_elements ||
            num_sources > gaussian_a->num_elements ||
            num_sources > gaussian_b->num_elements ||
            num_sources > gaussian_c->num_elements)
    {
        return OSKAR_ERR_DIMENSION_MISMATCH;
    }

    /* Return if memory is not on the CPU. */
    if (gaussian_a->location != OSKAR_LOCATION_CPU ||
            gaussian_b->location != OSKAR_LOCATION_CPU ||
            gaussian_c->location != OSKAR_LOCATION_CPU ||
            FWHM_major->location != OSKAR_LOCATION_CPU ||
            FWHM_minor->location != OSKAR_LOCATION_CPU ||
            position_angle->location != OSKAR_LOCATION_CPU)
    {
        return OSKAR_ERR_BAD_LOCATION;
    }

    if (type == OSKAR_DOUBLE)
    {
        for (i = 0; i < num_sources; ++i)
        {
            ra  = ((double*)RA->data)[i];
            dec = ((double*)Dec->data)[i];
            fwhm_maj = ((double*)FWHM_major->data)[i];
            fwhm_min = ((double*)FWHM_minor->data)[i];
            pa       = ((double*)position_angle->data)[i];

//            sin_pa = sin(pa);
//            cos_pa = cos(pa);
//
//            delta_ra_maj  = fwhm_maj/2.0 * sin_pa;
//            delta_dec_maj = fwhm_maj/2.0 * cos_pa;
//            delta_ra_min  = fwhm_min/2.0 * cos_pa;
//            delta_dec_min = fwhm_min/2.0 * sin_pa;
//
//            lon[0] = ra - delta_ra_maj;
//            lon[1] = ra + delta_ra_maj;
//            lon[2] = ra - delta_ra_min;
//            lon[3] = ra + delta_ra_min;
//
//            lat[0] = dec - delta_dec_maj;
//            lat[1] = dec + delta_dec_maj;
//            lat[2] = dec - delta_dec_min;
//            lat[3] = dec + delta_dec_min;
//
//            oskar_sph_from_lm_d(4, ra0, dec0, (double*)l.data, (double*)m.data,
//                    (double*)lon.data, (double*)lat.data);

//            delta_l_maj = MAX(l[0], l[1]) - MIN(l[0], l[1]);
//            delta_l_min = MAX(l[2], l[3]) - MIN(l[2], l[3]);
//            delta_m_maj = MAX(m[0], m[1]) - MIN(m[0], m[1]);
//            delta_m_min = MAX(m[2], m[3]) - MIN(m[2], m[3]);
//
//            pa_lm = atan2(delta_l_maj/2.0, delta_m_maj/2.0);




            inv_std_maj_2 = (fwhm_maj * fwhm_maj) * M_PI_2_2_LN_2;
            inv_std_min_2 = (fwhm_min * fwhm_min) * M_PI_2_2_LN_2;
            cos_pa_2 = cos(pa) * cos(pa);
            sin_pa_2 = sin(pa) * sin(pa);
            sin_2pa  = sin(2.0 * pa);
            a =  (cos_pa_2 * inv_std_min_2) / 2.0 + (sin_pa_2 * inv_std_maj_2) / 2.0;
            b = -(sin_2pa  * inv_std_min_2) / 4.0 + (sin_2pa  * inv_std_maj_2) / 4.0;
            c =  (sin_pa_2 * inv_std_min_2) / 2.0 + (cos_pa_2 * inv_std_maj_2) / 2.0;
            ((double*)gaussian_a->data)[i] = a;
            ((double*)gaussian_b->data)[i] = b;
            ((double*)gaussian_c->data)[i] = c;
        }
    }
    else
    {
        for (i = 0; i < num_sources; ++i)
        {
            // TODO: Convert to lm plane gaussian coordinates.

            fwhm_maj = ((float*)FWHM_major->data)[i];
            fwhm_min = ((float*)FWHM_minor->data)[i];
            pa       = ((float*)position_angle->data)[i];
            inv_std_maj_2 = (fwhm_maj * fwhm_maj) * M_PI_2_2_LN_2;
            inv_std_min_2 = (fwhm_min * fwhm_min) * M_PI_2_2_LN_2;
            cos_pa_2 = cos(pa) * cos(pa);
            sin_pa_2 = sin(pa) * sin(pa);
            sin_2pa  = sin(2.0 * pa);
            a =  (cos_pa_2 * inv_std_min_2) / 2.0 + (sin_pa_2 * inv_std_maj_2) / 2.0;
            b = -(sin_2pa  * inv_std_min_2) / 4.0 + (sin_2pa  * inv_std_maj_2) / 4.0;
            c =  (sin_pa_2 * inv_std_min_2) / 2.0 + (cos_pa_2 * inv_std_maj_2) / 2.0;
            ((float*)gaussian_a->data)[i] = (float)a;
            ((float*)gaussian_b->data)[i] = (float)b;
            ((float*)gaussian_c->data)[i] = (float)c;
        }
    }

    return OSKAR_SUCCESS;
}


#ifdef __cplusplus
}
#endif
