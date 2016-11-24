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

#include "imager/private_imager.h"
#include "imager/private_imager_free_dft.h"
#include "imager/private_imager_init_dft.h"
#include "math/oskar_evaluate_image_lmn_grid.h"
#include "math/oskar_cmath.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_init_dft(oskar_Imager* h, int* status)
{
    int num_pixels;
    oskar_imager_free_dft(h, status);
    if (*status) return;

    /* Calculate pixel coordinate grid required for the DFT imager. */
    num_pixels = h->image_size * h->image_size;
    h->l = oskar_mem_create(h->imager_prec, OSKAR_CPU, num_pixels, status);
    h->m = oskar_mem_create(h->imager_prec, OSKAR_CPU, num_pixels, status);
    h->n = oskar_mem_create(h->imager_prec, OSKAR_CPU, num_pixels, status);
    oskar_evaluate_image_lmn_grid(h->image_size, h->image_size,
            h->fov_deg * M_PI/180, h->fov_deg * M_PI/180, 0,
            h->l, h->m, h->n, status);
    oskar_mem_add_real(h->n, -1.0, status); /* n-1 */
}

#ifdef __cplusplus
}
#endif
