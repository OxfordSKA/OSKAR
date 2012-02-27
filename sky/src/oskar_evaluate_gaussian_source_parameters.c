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

#ifndef M_SQRT_2_LN2_PI
#define M_SQRT_2_LN2_PI 3.74781250258555160845195e-1 /* sqrt(2 * log_e(2)) / pi */
#endif

#ifdef __cplusplus
extern "C" {
#endif

int oskar_evaluate_gaussian_source_parameters(int num_sources,
        oskar_Mem* gaussian_a, oskar_Mem* gaussian_b, oskar_Mem* gaussian_c,
        oskar_Mem* FWHM_major, oskar_Mem* FWHM_minor, oskar_Mem* position_angle)
{
    int i;
    double a, b, c;
    double fwhm_maj, fwhm_min;
    double cos_pa_2, sin_pa_2, sin_2pa, std_maj, std_min, pa;
    double std_min_2, std_maj_2;

    if (gaussian_a == NULL || gaussian_b == NULL || gaussian_c == NULL ||
            FWHM_major == NULL || FWHM_minor == NULL || position_angle == NULL)
    {
        return OSKAR_ERR_INVALID_ARGUMENT;
    }

    if (num_sources != FWHM_major->private_num_elements ||
            num_sources != FWHM_minor->private_num_elements ||
            num_sources != position_angle->private_num_elements ||
            num_sources != gaussian_a->private_num_elements ||
            num_sources != gaussian_b->private_num_elements ||
            num_sources != gaussian_c->private_num_elements)
    {
        return OSKAR_ERR_DIMENSION_MISMATCH;
    }

    /* Return if memory is not on the CPU. */
    if (gaussian_a->private_location != OSKAR_LOCATION_CPU ||
            gaussian_b->private_location != OSKAR_LOCATION_CPU ||
            gaussian_c->private_location != OSKAR_LOCATION_CPU ||
            FWHM_major->private_location != OSKAR_LOCATION_CPU ||
            FWHM_minor->private_location != OSKAR_LOCATION_CPU ||
            position_angle->private_location != OSKAR_LOCATION_CPU)
    {
        return OSKAR_ERR_BAD_LOCATION;
    }

    if (gaussian_a->private_type == OSKAR_DOUBLE ||
            gaussian_b->private_type == OSKAR_DOUBLE ||
            gaussian_c->private_type == OSKAR_DOUBLE ||
            FWHM_major->private_type == OSKAR_DOUBLE ||
            FWHM_minor->private_type == OSKAR_DOUBLE ||
            position_angle->private_type == OSKAR_DOUBLE)
    {
        for (i = 0; i < num_sources; ++i)
        {
            fwhm_maj = ((double*)FWHM_major->data)[i];
            fwhm_min = ((double*)FWHM_minor->data)[i];

            pa = ((double*)position_angle->data)[i];

            std_maj = M_SQRT_2_LN2_PI / fwhm_maj;
            std_min = M_SQRT_2_LN2_PI / fwhm_min;
            printf("\n---- maj = %f, min = %f, pa = %f\n", std_maj, std_min, pa);
            std_maj_2 = std_maj * std_maj;
            std_min_2 = std_min * std_min;
            cos_pa_2 = cos(pa) * cos(pa);
            sin_pa_2 = sin(pa) * sin(pa);
            sin_2pa  = sin(2.0 * pa);
            a =  cos_pa_2 / (2.0 * std_min_2) + sin_pa_2 / (2.0 * std_maj_2);
            b = -sin_2pa  / (4.0 * std_min_2) + sin_2pa  / (4.0 * std_maj_2);
            c =  sin_pa_2 / (2.0 * std_min_2) + cos_pa_2 / (2.0 * std_maj_2);

            ((double*)gaussian_a->data)[i] = a;
            ((double*)gaussian_b->data)[i] = b;
            ((double*)gaussian_c->data)[i] = c;
        }
    }
    else if (gaussian_a->private_type == OSKAR_SINGLE ||
            gaussian_b->private_type == OSKAR_SINGLE ||
            gaussian_c->private_type == OSKAR_SINGLE ||
            FWHM_major->private_type == OSKAR_SINGLE ||
            FWHM_minor->private_type == OSKAR_SINGLE ||
            position_angle->private_type == OSKAR_SINGLE)
    {
        for (i = 0; i < num_sources; ++i)
        {
            fwhm_maj = ((float*)FWHM_major->data)[i];
            fwhm_min = ((float*)FWHM_minor->data)[i];
            pa       = ((float*)position_angle->data)[i];

            std_maj = M_SQRT_2_LN2_PI / fwhm_maj;
            std_min = M_SQRT_2_LN2_PI / fwhm_min;
            std_maj_2 = std_maj * std_maj;
            std_min_2 = std_min * std_min;
            cos_pa_2 = cos(pa) * cos(pa);
            sin_pa_2 = sin(pa) * sin(pa);
            sin_2pa  = sin(2.0 * pa);
            a =  cos_pa_2 / (2.0 * std_min_2) + sin_pa_2 / (2.0 * std_maj_2);
            b = -sin_2pa  / (4.0 * std_min_2) + sin_2pa  / (4.0 * std_maj_2);
            c =  sin_pa_2 / (2.0 * std_min_2) + cos_pa_2 / (2.0 * std_maj_2);

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
