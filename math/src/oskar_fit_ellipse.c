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


#include "math/oskar_fit_ellipse.h"
#include "math/oskar_mean.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_free.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_fit_ellipse(double* a, double* b, double* phi,
        int num_points, const oskar_Mem* x, const oskar_Mem* y)
{
    int i, type, location;
    double orientation_tolerance;
    double mean_x, mean_y;
    oskar_Mem x2, y2, X;

    if (x->type == OSKAR_DOUBLE && y->type == OSKAR_DOUBLE)
        type = OSKAR_DOUBLE;
    else if (x->type == OSKAR_SINGLE && y->type == OSKAR_SINGLE)
        type = OSKAR_SINGLE;
    else
        return OSKAR_ERR_BAD_DATA_TYPE;

    location = OSKAR_LOCATION_CPU;

    orientation_tolerance = 1.0e-3;

    /* remove bias of ellipse */
    oskar_mean(&mean_x, num_points, x);
    oskar_mean(&mean_y, num_points, y);
    oskar_mem_init(&x2, type, location, num_points, OSKAR_TRUE);
    oskar_mem_init(&y2, type, location, num_points, OSKAR_TRUE);
    if (type == OSKAR_DOUBLE)
    {
        for (i = 0; i < num_points; ++i)
        {
            ((double*)x2->data)[i] = ((double*)x->data)[i] - mean_x;
            ((double*)y2->data)[i] = ((double*)y->data)[i] - mean_y;
        }
    }
    else
    {
        for (i = 0; i < num_points; ++i)
        {
            ((float*)x2->data)[i] = ((float*)x->data)[i] - mean_x;
            ((float*)y2->data)[i] = ((float*)y->data)[i] - mean_y;
        }
    }


    /* estimation of the conic equation of the ellipse */
    oskar_mem_init(&X, type, location, 5 * num_points);
    /* TODO construct X */
    /* TODO result = sum(X) / (X' * X) */


    /* TODO extract parameters from conic equation */
    /* ... */

    /* clean up */
    oskar_mem_free(&x2);
    oskar_mem_free(&y2);
    oskar_mem_free(&X);

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
