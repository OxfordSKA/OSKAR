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

#include "sky/oskar_evaluate_gaussian_source_parameters.h"
#include "utility/oskar_log_warning.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_free.h"
#include "math/oskar_sph_to_lm.h"
#include "math/oskar_sph_from_lm.h"
#include "math/oskar_sph_rotate_points.h"
#include "math/oskar_fit_ellipse.h"

#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define M_PI_2_2_LN_2 7.11941466249375271693034 /* pi^2 / (2 log_e(2)) */
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#ifdef __cplusplus
extern "C" {
#endif

/* FIXME HACK 'I' added as hack */
int oskar_evaluate_gaussian_source_parameters(oskar_Log* log, int num_sources,
        oskar_Mem* gaussian_a, oskar_Mem* gaussian_b, oskar_Mem* gaussian_c,
        const oskar_Mem* FWHM_major, const oskar_Mem* FWHM_minor,
        const oskar_Mem* position_angle, const oskar_Mem* RA,
        const oskar_Mem* Dec, oskar_Mem* I, double ra0, double dec0)
{
    int i, j, err, num_failed = 0;
    double a, b, c;
    int type;
    double maj, min, pa;
    double cos_pa_2, sin_pa_2, sin_2pa;
    double inv_std_min_2, inv_std_maj_2;
    double ra, dec;
    double ellipse_a, ellipse_b;

    oskar_Mem l, m;
    int ellipse_points;
    oskar_Mem lon, lat;

    err = OSKAR_SUCCESS;

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

    ellipse_points = 360/60;
    oskar_mem_init(&l, type, OSKAR_LOCATION_CPU, ellipse_points, OSKAR_TRUE, &err);
    oskar_mem_init(&m, type, OSKAR_LOCATION_CPU, ellipse_points, OSKAR_TRUE, &err);
    oskar_mem_init(&lon, type, OSKAR_LOCATION_CPU, ellipse_points, OSKAR_TRUE, &err);
    oskar_mem_init(&lat, type, OSKAR_LOCATION_CPU, ellipse_points, OSKAR_TRUE, &err);
    if (err) return err;

    if (type == OSKAR_DOUBLE)
    {
        for (i = 0; i < num_sources; ++i)
        {
            /* Source parameters */
            ra  = ((double*)RA->data)[i];
            dec = ((double*)Dec->data)[i];
            maj = ((double*)FWHM_major->data)[i];
            min = ((double*)FWHM_minor->data)[i];
            pa  = ((double*)position_angle->data)[i];

            /* Note: could maybe do something different from the projection below
             * in the case of a line (ie maj or min == 0) as in this case
             * there is no ellipse to project only two points....
             * -- This continue could then be a if() .. else() instead.
             */
            if (maj == 0.0 && min == 0.0) continue;

            /* Evaluate shape of ellipse on the lm plane */
            ellipse_a = maj/2.0;
            ellipse_b = min/2.0;
            for (j = 0; j < ellipse_points; ++j)
            {
                double t = (double)j * 60.0 * M_PI/180.0;
                ((double*)l.data)[j] = ellipse_a*cos(t)*sin(pa) + ellipse_b*sin(t)*cos(pa);
                ((double*)m.data)[j] = ellipse_a*cos(t)*cos(pa) - ellipse_b*sin(t)*sin(pa);
            }
            oskar_sph_from_lm_d(ellipse_points, 0.0, 0.0,
                    (double*)l.data, (double*)m.data,
                    (double*)lon.data, (double*)lat.data);
            err = oskar_sph_rotate_points(ellipse_points, &lon, &lat, ra, dec);
            if (err) return err;
            oskar_sph_to_lm_d(ellipse_points, ra0, dec0,
                    (double*)lon.data, (double*)lat.data,
                    (double*)l.data, (double*)m.data);
            err = oskar_fit_ellipse(log, &maj, &min, &pa,
                    ellipse_points, &l, &m);
            if (err == OSKAR_ERR_ELLIPSE_FIT_FAILED)
            {
                /* FIXME HACK */
                ((double*)I->data)[i] = 0.0;
                ++num_failed;
                continue;
            }
            else if (err) return err;

            /* Evaluate ellipse parameters */
            inv_std_maj_2 = (maj * maj) * M_PI_2_2_LN_2;
            inv_std_min_2 = (min * min) * M_PI_2_2_LN_2;
            cos_pa_2 = cos(pa) * cos(pa);
            sin_pa_2 = sin(pa) * sin(pa);
            sin_2pa  = sin(2.0 * pa);
            a =  (cos_pa_2*inv_std_min_2)/2.0 + (sin_pa_2*inv_std_maj_2)/2.0;
            b = -(sin_2pa *inv_std_min_2)/4.0 + (sin_2pa *inv_std_maj_2)/4.0;
            c =  (sin_pa_2*inv_std_min_2)/2.0 + (cos_pa_2*inv_std_maj_2)/2.0;
            ((double*)gaussian_a->data)[i] = a;
            ((double*)gaussian_b->data)[i] = b;
            ((double*)gaussian_c->data)[i] = c;
        }
    }
    else
    {
        for (i = 0; i < num_sources; ++i)
         {
             /* Source parameters */
             ra  = ((float*)RA->data)[i];
             dec = ((float*)Dec->data)[i];
             maj = ((float*)FWHM_major->data)[i];
             min = ((float*)FWHM_minor->data)[i];
             pa  = ((float*)position_angle->data)[i];

             /* Note: could maybe do something different from the projection below
              * in the case of a line (ie maj or min == 0) as in this case
              * there is no ellipse to project only two points....
              * -- This continue could then be a if() .. else() instead.
              */
             if (maj == 0.0 && min == 0.0) continue;

             /* Evaluate shape of ellipse on the lm plane */
             ellipse_a = maj/2.0;
             ellipse_b = min/2.0;
             for (j = 0; j < ellipse_points; ++j)
             {
                 float t = (float)j * 60.0 * M_PI/180.0;
                 ((float*)l.data)[j] = ellipse_a*cosf(t)*sinf(pa) + ellipse_b*sinf(t)*cosf(pa);
                 ((float*)m.data)[j] = ellipse_a*cosf(t)*cosf(pa) - ellipse_b*sinf(t)*sinf(pa);
             }
             oskar_sph_from_lm_f(ellipse_points, 0.0, 0.0,
                     (float*)l.data, (float*)m.data,
                     (float*)lon.data, (float*)lat.data);
             err = oskar_sph_rotate_points(ellipse_points, &lon, &lat, ra, dec);
             if (err) return err;
             oskar_sph_to_lm_f(ellipse_points, ra0, dec0,
                     (float*)lon.data, (float*)lat.data,
                     (float*)l.data, (float*)m.data);
             err = oskar_fit_ellipse(log, &maj, &min, &pa,
                     ellipse_points, &l, &m);
             if (err == OSKAR_ERR_ELLIPSE_FIT_FAILED)
             {
                 /* FIXME HACK */
                 ((float*)I->data)[i] = 0.0;
                 ++num_failed;
                 continue;
             }
             else if (err) return err;

             /* Evaluate ellipse parameters */
             inv_std_maj_2 = (maj * maj) * M_PI_2_2_LN_2;
             inv_std_min_2 = (min * min) * M_PI_2_2_LN_2;
             cos_pa_2 = cosf(pa) * cosf(pa);
             sin_pa_2 = sinf(pa) * sinf(pa);
             sin_2pa  = sinf(2.0 * pa);
             a =  (cos_pa_2*inv_std_min_2)/2.0 + (sin_pa_2*inv_std_maj_2)/2.0;
             b = -(sin_2pa *inv_std_min_2)/4.0 + (sin_2pa *inv_std_maj_2)/4.0;
             c =  (sin_pa_2*inv_std_min_2)/2.0 + (cos_pa_2*inv_std_maj_2)/2.0;
             ((float*)gaussian_a->data)[i] = a;
             ((float*)gaussian_b->data)[i] = b;
             ((float*)gaussian_c->data)[i] = c;
         }
    }

    /* Print a warning if fitting failed for some sources. */
    if (num_failed > 0)
    {
        oskar_log_warning(log, "Gaussian ellipse solution failed "
                "for %i sources. These sources will be simulated as point objects.",
                num_failed);
    }

    /* clean up */
    oskar_mem_free(&l);
    oskar_mem_free(&m);
    oskar_mem_free(&lon);
    oskar_mem_free(&lat);

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
