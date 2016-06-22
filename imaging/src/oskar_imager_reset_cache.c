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
#include <private_imager_free_dft.h>
#include <private_imager_free_fft.h>
#include <private_imager_free_wproj.h>
#include <oskar_imager_reset_cache.h>
#include <fitsio.h>

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_reset_cache(oskar_Imager* h, int* status)
{
    int i;
    oskar_imager_free_dft(h, status);
    oskar_imager_free_fft(h, status);
    oskar_imager_free_wproj(h, status);
    free(h->plane_norm);
    for (i = 0; i < h->num_planes; ++i)
        oskar_mem_free(h->planes[i], status);
    free(h->planes);
    oskar_mem_realloc(h->uu_im, 0, status);
    oskar_mem_realloc(h->vv_im, 0, status);
    oskar_mem_realloc(h->ww_im, 0, status);
    oskar_mem_realloc(h->uu_tmp, 0, status);
    oskar_mem_realloc(h->vv_tmp, 0, status);
    oskar_mem_realloc(h->ww_tmp, 0, status);
    oskar_mem_realloc(h->vis_im, 0, status);
    oskar_mem_realloc(h->weight_im, 0, status);
    oskar_mem_realloc(h->weight_tmp, 0, status);
    oskar_mem_free(h->stokes, status);
    for (i = 0; i < h->im_num_pols; ++i)
    {
        if (h->fits_file[i])
            ffclos(h->fits_file[i], status);
        h->fits_file[i] = 0;
    }
    h->plane_norm = 0;
    h->num_planes = 0;
    h->planes = 0;
    h->stokes = 0;
}

#ifdef __cplusplus
}
#endif
