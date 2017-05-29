/*
 * Copyright (c) 2016-2017, The University of Oxford
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

#ifdef OSKAR_HAVE_CUDA
#include <cufft.h>
#endif

#include "imager/private_imager.h"
#include "imager/oskar_imager_reset_cache.h"
#include <fitsio.h>

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_reset_cache(oskar_Imager* h, int* status)
{
    int i;

    /* Clear selected axes. */
    free(h->sel_freqs);
    free(h->im_freqs);
    h->sel_freqs = 0;
    h->im_freqs = 0;
    h->num_sel_freqs = 0;
    h->num_im_channels = 0;

    /* Clear FFT caches. */
    oskar_mem_free(h->corr_func, status);
    oskar_mem_free(h->fftpack_wsave, status);
    oskar_mem_free(h->fftpack_work, status);
#ifdef OSKAR_HAVE_CUDA
    cufftDestroy(h->cufft_plan);
    h->cufft_plan = 0;
#endif
    h->corr_func = 0;
    h->fftpack_wsave = 0;
    h->fftpack_work = 0;

    /* Clear algorithm-specific caches. */
    oskar_mem_free(h->l, status); h->l = 0;
    oskar_mem_free(h->m, status); h->m = 0;
    oskar_mem_free(h->n, status); h->n = 0;
    oskar_mem_free(h->conv_func, status); h->conv_func = 0;
    oskar_mem_free(h->w_kernels, status); h->w_kernels = 0;
    oskar_mem_free(h->w_support, status); h->w_support = 0;

    /* Free the image planes. */
    if (h->planes)
        for (i = 0; i < h->num_planes; ++i)
            oskar_mem_free(h->planes[i], status);
    free(h->planes);
    h->planes = 0;
    free(h->plane_norm);
    h->plane_norm = 0;

    /* Free the weights grids if they exist. */
    if (h->weights_grids)
        for (i = 0; i < h->num_planes; ++i)
            oskar_mem_free(h->weights_grids[i], status);
    free(h->weights_grids);
    h->weights_grids = 0;

    /* Collapse temp arrays. */
    oskar_mem_realloc(h->uu_im, 0, status);
    oskar_mem_realloc(h->vv_im, 0, status);
    oskar_mem_realloc(h->ww_im, 0, status);
    oskar_mem_realloc(h->uu_tmp, 0, status);
    oskar_mem_realloc(h->vv_tmp, 0, status);
    oskar_mem_realloc(h->ww_tmp, 0, status);
    oskar_mem_realloc(h->vis_im, 0, status);
    oskar_mem_realloc(h->weight_im, 0, status);
    oskar_mem_realloc(h->weight_tmp, 0, status);
    oskar_mem_realloc(h->time_im, 0, status);
    oskar_mem_free(h->stokes, status);
    h->stokes = 0;

    /* Close any open FITS files. */
    for (i = 0; i < h->num_im_pols; ++i)
    {
        if (h->fits_file[i])
            ffclos(h->fits_file[i], status);
        h->fits_file[i] = 0;
        free(h->output_name[i]);
        h->output_name[i] = 0;
    }

    /* Clear the number of image planes. */
    h->num_planes = 0;
}

#ifdef __cplusplus
}
#endif
