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

#include "math/oskar_spline_data_compute.h"
#include "math/oskar_spline_data_init.h"
#include "math/oskar_spline_set_up.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_realloc.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_spline_data_compute(oskar_SplineData* spline, int num_x, int num_y,
        double start_x, double start_y, double end_x, double end_y,
        const oskar_Mem* data)
{
    int type, err, i;
    oskar_Mem x_axis, y_axis;

    /* Get the data type. */
    type = data->private_type;
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
        return OSKAR_ERR_BAD_DATA_TYPE;

    /* Check that input data is on the CPU. */
    if (data->private_location != OSKAR_LOCATION_CPU)
        return OSKAR_ERR_BAD_LOCATION;

    /* Initialise and allocate spline data. */
    err = oskar_spline_data_init(spline, type, OSKAR_LOCATION_CPU);
    if (err) return err;
    spline->degree_x = 3;
    spline->degree_y = 3;
    err = oskar_mem_realloc(&spline->knots_x, 1 + spline->degree_x + num_x);
    if (err) return err;
    err = oskar_mem_realloc(&spline->knots_y, 1 + spline->degree_y + num_y);
    if (err) return err;
    err = oskar_mem_realloc(&spline->coeff, num_x * num_y);
    if (err) return err;

    /* Initialise axis arrays. */
    oskar_mem_init(&x_axis, type, OSKAR_LOCATION_CPU, 0, OSKAR_TRUE);
    oskar_mem_init(&y_axis, type, OSKAR_LOCATION_CPU, 0, OSKAR_TRUE);

    /* Allocate axis arrays. */
    err = oskar_mem_init(&x_axis, type, OSKAR_LOCATION_CPU, num_x, OSKAR_TRUE);
    if (err) goto stop;
    err = oskar_mem_init(&y_axis, type, OSKAR_LOCATION_CPU, num_y, OSKAR_TRUE);
    if (err) goto stop;

    if (type == OSKAR_SINGLE)
    {
        float *x, *y, *z, *tx, *ty, *c;
        x =  (float*)x_axis.data;
        y =  (float*)y_axis.data;
        z =  (float*)data->data;
        tx = (float*)spline->knots_x.data;
        ty = (float*)spline->knots_y.data;
        c =  (float*)spline->coeff.data;

        /* Create the data axes. */
        for (i = 0; i < num_x; ++i)
            x[i] = i * (float)((end_x - start_x) / (num_x - 1));
        for (i = 0; i < num_y; ++i)
            y[i] = i * (float)((end_y - start_y) / (num_y - 1));

        /* Set up the spline data. */
        err = oskar_spline_set_up_f(num_x, x, num_y, y, z, spline->degree_x,
                tx, spline->degree_y, ty, &spline->num_knots_x,
                &spline->num_knots_y, c);
    }
    else if (type == OSKAR_DOUBLE)
    {
        double *x, *y, *z, *tx, *ty, *c;
        x =  (double*)x_axis.data;
        y =  (double*)y_axis.data;
        z =  (double*)data->data;
        tx = (double*)spline->knots_x.data;
        ty = (double*)spline->knots_y.data;
        c =  (double*)spline->coeff.data;

        /* Create the data axes. */
        for (i = 0; i < num_x; ++i)
            x[i] = i * (end_x - start_x) / (num_x - 1);
        for (i = 0; i < num_y; ++i)
            y[i] = i * (end_y - start_y) / (num_y - 1);

        /* Set up the spline data. */
        err = OSKAR_ERR_UNKNOWN;
    }

    /* Free axis arrays. */
    stop:
    oskar_mem_free(&x_axis);
    oskar_mem_free(&y_axis);
    return err;
}

#ifdef __cplusplus
}
#endif
