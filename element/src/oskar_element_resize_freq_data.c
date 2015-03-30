/*
 * Copyright (c) 2014-2015, The University of Oxford
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

#include <private_element.h>
#include <oskar_element.h>

#ifdef __cplusplus
extern "C" {
#endif

static void realloc_arrays(oskar_Element* e, int size, int* status);

void oskar_element_resize_freq_data(oskar_Element* model, int size,
        int* status)
{
    int i, old_size, loc, precision;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the old size. */
    old_size = model->num_freq;
    loc = oskar_element_mem_location(model);
    precision = oskar_element_precision(model);

    if (size > old_size)
    {
        /* Enlarge the arrays and create new structures. */
        realloc_arrays(model, size, status);
        for (i = old_size; i < size; ++i)
        {
            model->filename_x[i] = oskar_mem_create(OSKAR_CHAR,
                    OSKAR_CPU, 0, status);
            model->filename_y[i] = oskar_mem_create(OSKAR_CHAR,
                    OSKAR_CPU, 0, status);
            model->filename_scalar[i] = oskar_mem_create(OSKAR_CHAR,
                    OSKAR_CPU, 0, status);
            model->x_v_re[i] = oskar_splines_create(precision, loc, status);
            model->x_v_im[i] = oskar_splines_create(precision, loc, status);
            model->x_h_re[i] = oskar_splines_create(precision, loc, status);
            model->x_h_im[i] = oskar_splines_create(precision, loc, status);
            model->y_v_re[i] = oskar_splines_create(precision, loc, status);
            model->y_v_im[i] = oskar_splines_create(precision, loc, status);
            model->y_h_re[i] = oskar_splines_create(precision, loc, status);
            model->y_h_im[i] = oskar_splines_create(precision, loc, status);
            model->scalar_re[i] = oskar_splines_create(precision, loc, status);
            model->scalar_im[i] = oskar_splines_create(precision, loc, status);
        }
    }
    else if (size < old_size)
    {
        /* Free old structures and shrink the arrays. */
        for (i = size; i < old_size; ++i)
        {
            oskar_mem_free(model->filename_x[i], status);
            oskar_mem_free(model->filename_y[i], status);
            oskar_mem_free(model->filename_scalar[i], status);
            oskar_splines_free(model->x_v_re[i], status);
            oskar_splines_free(model->x_v_im[i], status);
            oskar_splines_free(model->x_h_re[i], status);
            oskar_splines_free(model->x_h_im[i], status);
            oskar_splines_free(model->y_v_re[i], status);
            oskar_splines_free(model->y_v_im[i], status);
            oskar_splines_free(model->y_h_re[i], status);
            oskar_splines_free(model->y_h_im[i], status);
            oskar_splines_free(model->scalar_re[i], status);
            oskar_splines_free(model->scalar_im[i], status);
        }
        realloc_arrays(model, size, status);
    }
    else
    {
        /* No resize necessary. */
        return;
    }

    /* Store the new size. */
    model->num_freq = size;
}

static void realloc_arrays(oskar_Element* e, int size, int* status)
{
    e->freqs_hz = realloc(e->freqs_hz, size * sizeof(double));
    if (!e->freqs_hz) *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
    e->filename_x = realloc(e->filename_x, size * sizeof(oskar_Mem*));
    if (!e->filename_x) *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
    e->filename_y = realloc(e->filename_y, size * sizeof(oskar_Mem*));
    if (!e->filename_y) *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
    e->filename_scalar = realloc(e->filename_scalar, size * sizeof(oskar_Mem*));
    if (!e->filename_scalar) *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
    e->x_v_re = realloc(e->x_v_re, size * sizeof(oskar_Splines*));
    if (!e->x_v_re) *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
    e->x_v_im = realloc(e->x_v_im, size * sizeof(oskar_Splines*));
    if (!e->x_v_im) *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
    e->x_h_re = realloc(e->x_h_re, size * sizeof(oskar_Splines*));
    if (!e->x_h_re) *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
    e->x_h_im = realloc(e->x_h_im, size * sizeof(oskar_Splines*));
    if (!e->x_h_im) *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
    e->y_v_re = realloc(e->y_v_re, size * sizeof(oskar_Splines*));
    if (!e->y_v_re) *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
    e->y_v_im = realloc(e->y_v_im, size * sizeof(oskar_Splines*));
    if (!e->y_v_im) *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
    e->y_h_re = realloc(e->y_h_re, size * sizeof(oskar_Splines*));
    if (!e->y_h_re) *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
    e->y_h_im = realloc(e->y_h_im, size * sizeof(oskar_Splines*));
    if (!e->y_h_im) *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
    e->scalar_re = realloc(e->scalar_re, size * sizeof(oskar_Splines*));
    if (!e->scalar_re) *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
    e->scalar_im = realloc(e->scalar_im, size * sizeof(oskar_Splines*));
    if (!e->scalar_im) *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
}

#ifdef __cplusplus
}
#endif
