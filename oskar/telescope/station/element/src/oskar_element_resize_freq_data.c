/*
 * Copyright (c) 2014-2019, The University of Oxford
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

#include "telescope/station/element/private_element.h"
#include "telescope/station/element/oskar_element.h"

#ifdef __cplusplus
extern "C" {
#endif

static void realloc_arrays(oskar_Element* e, int size);

void oskar_element_resize_freq_data(oskar_Element* model, int size,
        int* status)
{
    int i;
    if (*status) return;
    const int old_size = model->num_freq;
    if (size > old_size)
    {
        /* Enlarge the arrays and create new structures. */
        realloc_arrays(model, size);
        for (i = old_size; i < size; ++i)
        {
            model->filename_x[i] = 0;
            model->filename_y[i] = 0;
            model->filename_scalar[i] = 0;
            model->x_v_re[i] = 0;
            model->x_v_im[i] = 0;
            model->x_h_re[i] = 0;
            model->x_h_im[i] = 0;
            model->y_v_re[i] = 0;
            model->y_v_im[i] = 0;
            model->y_h_re[i] = 0;
            model->y_h_im[i] = 0;
            model->scalar_re[i] = 0;
            model->scalar_im[i] = 0;
            model->sph_wave[i] = 0;
            model->l_max[i] = 0;
            model->common_phi_coords[i] = 0;
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
            oskar_mem_free(model->sph_wave[i], status);

        }
        realloc_arrays(model, size);
    }
    model->num_freq = size;
}

static void realloc_arrays(oskar_Element* e, int size)
{
    const size_t sz = size * sizeof(void*);
    e->freqs_hz = (double*) realloc(e->freqs_hz, size * sizeof(double));
    e->l_max = (int*) realloc(e->l_max, size * sizeof(int));
    e->common_phi_coords = (int*) realloc(
            e->common_phi_coords, size * sizeof(int));
    e->filename_x = (oskar_Mem**) realloc(e->filename_x, sz);
    e->filename_y = (oskar_Mem**) realloc(e->filename_y, sz);
    e->filename_scalar = (oskar_Mem**) realloc(e->filename_scalar, sz);
    e->x_v_re = (oskar_Splines**) realloc(e->x_v_re, sz);
    e->x_v_im = (oskar_Splines**) realloc(e->x_v_im, sz);
    e->x_h_re = (oskar_Splines**) realloc(e->x_h_re, sz);
    e->x_h_im = (oskar_Splines**) realloc(e->x_h_im, sz);
    e->y_v_re = (oskar_Splines**) realloc(e->y_v_re, sz);
    e->y_v_im = (oskar_Splines**) realloc(e->y_v_im, sz);
    e->y_h_re = (oskar_Splines**) realloc(e->y_h_re, sz);
    e->y_h_im = (oskar_Splines**) realloc(e->y_h_im, sz);
    e->scalar_re = (oskar_Splines**) realloc(e->scalar_re, sz);
    e->scalar_im = (oskar_Splines**) realloc(e->scalar_im, sz);
    e->sph_wave = (oskar_Mem**) realloc(e->sph_wave, sz);
}

#ifdef __cplusplus
}
#endif
