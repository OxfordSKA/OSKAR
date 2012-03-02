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
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define M_PI_2_2_LN_2 7.11941466249375271693034 /* pi^2 / (2 log_e(2)) */

#ifdef __cplusplus
extern "C" {
#endif

int oskar_evaluate_gaussian_source_parameters(int num_sources,
        oskar_Mem* gaussian_a, oskar_Mem* gaussian_b, oskar_Mem* gaussian_c,
        oskar_Mem* FWHM_major, oskar_Mem* FWHM_minor, oskar_Mem* position_angle)
{
    int i;
    double a, b, c;
    double fwhm_maj, fwhm_min, pa;
    double cos_pa_2, sin_pa_2, sin_2pa;
    double inv_std_min_2, inv_std_maj_2;

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

    if (gaussian_a->type == OSKAR_DOUBLE ||
            gaussian_b->type == OSKAR_DOUBLE ||
            gaussian_c->type == OSKAR_DOUBLE ||
            FWHM_major->type == OSKAR_DOUBLE ||
            FWHM_minor->type == OSKAR_DOUBLE ||
            position_angle->type == OSKAR_DOUBLE)
    {
        for (i = 0; i < num_sources; ++i)
        {
            fwhm_maj = ((double*)FWHM_major->data)[i];
            fwhm_min = ((double*)FWHM_minor->data)[i];
            pa       = ((double*)position_angle->data)[i];
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
    else if (gaussian_a->type == OSKAR_SINGLE ||
            gaussian_b->type == OSKAR_SINGLE ||
            gaussian_c->type == OSKAR_SINGLE ||
            FWHM_major->type == OSKAR_SINGLE ||
            FWHM_minor->type == OSKAR_SINGLE ||
            position_angle->type == OSKAR_SINGLE)
    {
        for (i = 0; i < num_sources; ++i)
        {
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
    else
    {
        return OSKAR_ERR_BAD_DATA_TYPE;
    }

    return OSKAR_SUCCESS;
}


#ifdef __cplusplus
}
#endif
