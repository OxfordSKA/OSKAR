/*
 * Copyright (c) 2016, The University of Oxford
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

#include <private_imager.h>

#include <oskar_imager_create.h>
#include <oskar_imager_set_options.h>

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Imager* oskar_imager_create(int imager_precision, int* status)
{
    oskar_Imager* h = 0;
    h = (oskar_Imager*) calloc(1, sizeof(oskar_Imager));

    /* Create scratch arrays. */
    h->imager_prec = imager_precision;
    h->uu_im       = oskar_mem_create(imager_precision, OSKAR_CPU, 0, status);
    h->vv_im       = oskar_mem_create(imager_precision, OSKAR_CPU, 0, status);
    h->ww_im       = oskar_mem_create(imager_precision, OSKAR_CPU, 0, status);
    h->uu_tmp      = oskar_mem_create(imager_precision, OSKAR_CPU, 0, status);
    h->vv_tmp      = oskar_mem_create(imager_precision, OSKAR_CPU, 0, status);
    h->ww_tmp      = oskar_mem_create(imager_precision, OSKAR_CPU, 0, status);
    h->vis_im      = oskar_mem_create(imager_precision | OSKAR_COMPLEX,
            OSKAR_CPU, 0, status);
    h->weight_im   = oskar_mem_create(imager_precision, OSKAR_CPU, 0, status);
    h->stokes      = oskar_mem_create(imager_precision | OSKAR_COMPLEX,
            OSKAR_CPU, 0, status);

    /* Set sensible defaults. */
    oskar_imager_set_gpus(h, -1, 0, status);
    oskar_imager_set_fft_on_gpu(h, 1);
    oskar_imager_set_size(h, 256);
    oskar_imager_set_fov(h, 1.0);
    oskar_imager_set_time_range(h, 0, -1, 0); /* Time synthesis. */
    oskar_imager_set_channel_range(h, 0, -1, 1); /* Channel snapshots. */
    oskar_imager_set_image_type(h, "I", status);
    oskar_imager_set_algorithm(h, "FFT", status);
    oskar_imager_set_ms_column(h, "DATA", status);
    oskar_imager_set_grid_kernel(h, "Spheroidal", 3, 100, status);
    oskar_imager_set_default_direction(h);
    return h;
}

#ifdef __cplusplus
}
#endif
