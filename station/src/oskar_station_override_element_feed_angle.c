/*
 * Copyright (c) 2015, The University of Oxford
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

#include <private_station.h>
#include <oskar_station.h>
#include <oskar_random_gaussian.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_override_element_feed_angle(oskar_Station* s,
        unsigned int seed, int x_pol, double alpha_error_rad,
        double beta_error_rad, double gamma_error_rad, int* status)
{
    int i;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check if there are child stations. */
    if (oskar_station_has_child(s))
    {
        /* Recursive call to find the last level (i.e. the element data). */
        for (i = 0; i < s->num_elements; ++i)
        {
            oskar_station_override_element_feed_angle(
                    oskar_station_child(s, i), seed, x_pol,
                    alpha_error_rad, beta_error_rad, gamma_error_rad, status);
        }
    }
    else
    {
        /* Override element data at last level. */
        int id;
        double *a, *b, *c, r[4];
        oskar_Mem *alpha, *beta, *gamma;

        /* Get pointer to the X or Y element orientation data. */
        alpha = x_pol ? s->element_x_alpha_cpu : s->element_y_alpha_cpu;
        beta  = x_pol ? s->element_x_beta_cpu : s->element_y_beta_cpu;
        gamma = x_pol ? s->element_x_gamma_cpu : s->element_y_gamma_cpu;
        a = oskar_mem_double(alpha, status);
        b = oskar_mem_double(beta, status);
        c = oskar_mem_double(gamma, status);
        id = oskar_station_unique_id(s);
        for (i = 0; i < s->num_elements; ++i)
        {
            /* Generate random numbers from Gaussian distribution. */
            oskar_random_gaussian4(seed, i, id, 0, 0, r);
            r[0] *= alpha_error_rad;
            r[1] *= beta_error_rad;
            r[2] *= gamma_error_rad;

            /* Set the new angle. */
            a[i] += r[0];
            b[i] += r[1];
            c[i] += r[2];
        }
    }
}

#ifdef __cplusplus
}
#endif
