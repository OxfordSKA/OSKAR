/*
 * Copyright (c) 2012-2015, The University of Oxford
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

void oskar_element_free(oskar_Element* data, int* status)
{
    int i;
    if (!data) return;

    /* Free the memory contents. */
    for (i = 0; i < data->num_freq; ++i)
    {
        oskar_mem_free(data->filename_x[i], status);
        oskar_mem_free(data->filename_y[i], status);
        oskar_mem_free(data->filename_scalar[i], status);
        oskar_splines_free(data->x_v_re[i], status);
        oskar_splines_free(data->x_v_im[i], status);
        oskar_splines_free(data->x_h_re[i], status);
        oskar_splines_free(data->x_h_im[i], status);
        oskar_splines_free(data->y_v_re[i], status);
        oskar_splines_free(data->y_v_im[i], status);
        oskar_splines_free(data->y_h_re[i], status);
        oskar_splines_free(data->y_h_im[i], status);
        oskar_splines_free(data->scalar_re[i], status);
        oskar_splines_free(data->scalar_im[i], status);
    }
    free(data->freqs_hz);
    free(data->filename_x);
    free(data->filename_y);
    free(data->filename_scalar);
    free(data->x_h_re);
    free(data->x_h_im);
    free(data->x_v_re);
    free(data->x_v_im);
    free(data->y_h_re);
    free(data->y_h_im);
    free(data->y_v_re);
    free(data->y_v_im);
    free(data->scalar_re);
    free(data->scalar_im);

    /* Free the structure itself. */
    free(data);
}

#ifdef __cplusplus
}
#endif
