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

#include <oskar_imager_free.h>
#include <oskar_imager_reset_cache.h>
#include <private_imager_free_gpu_data.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_free(oskar_Imager* h, int* status)
{
    if (!h) return;
    oskar_imager_reset_cache(h, status);
    oskar_mem_free(h->uu_im, status);
    oskar_mem_free(h->vv_im, status);
    oskar_mem_free(h->ww_im, status);
    oskar_mem_free(h->uu_tmp, status);
    oskar_mem_free(h->vv_tmp, status);
    oskar_mem_free(h->ww_tmp, status);
    oskar_mem_free(h->vis_im, status);
    oskar_mem_free(h->stokes, status);

    oskar_imager_free_gpu_data(h, status);
    free(h->image_root);
    free(h->ms_column);
    free(h);
}

#ifdef __cplusplus
}
#endif
