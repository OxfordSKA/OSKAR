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

#include "imaging/oskar_make_image_dft.h"
#include "math/oskar_cuda_dft_c2r_2d.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_mem_copy.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_scale_real.h"

#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if 0
int oskar_make_image_dft(oskar_Mem* image, const oskar_Mem* uu_metres,
        const oskar_Mem* vv_metres, const oskar_Mem* amp, const oskar_Mem* l,
        const oskar_Mem* m, double frequency_hz)
{
    oskar_Mem u, v, t_l, t_m, t_amp, t_image, *p_image;
    const oskar_Mem *p_l, *p_m, *p_amp;
    double wavenumber;
    int err, type, loc_image, loc_amp, loc_l, loc_m, num_vis, num_pixels;

    /* Check types. */
    type = image->type;
    if (type != uu_metres->type || type != vv_metres->type ||
            type != (amp->type | OSKAR_COMPLEX) ||
            type != l->type || type != m->type)
        return OSKAR_ERR_TYPE_MISMATCH;

    /* Get dimension sizes. */
    num_vis = amp->num_elements;
    num_pixels = l->num_elements;
    if (num_pixels != m->num_elements)
        return OSKAR_ERR_DIMENSION_MISMATCH;

    /* Get data locations. */
    loc_image = image->location;
    loc_amp = amp->location;
    loc_l = l->location;
    loc_m = m->location;

    /* Compute the wavenumber. */
    wavenumber = 2.0 * M_PI * frequency_hz / 299792458.0;

    /* Initialise all temporary array (to zero length). */
    oskar_mem_init(&u, uu_metres->type, OSKAR_LOCATION_GPU, 0, 1);
    oskar_mem_init(&v, vv_metres->type, OSKAR_LOCATION_GPU, 0, 1);
    oskar_mem_init(&t_l, vv_metres->type, OSKAR_LOCATION_GPU, 0, 1);

    /* Copy the baselines to temporary GPU memory. */
    err = oskar_mem_copy(&u, uu_metres);
    if (err) goto cleanup;
    err = oskar_mem_copy(&v, vv_metres);
    if (err) goto cleanup;

    /* Multiply baselines by the wavenumber. */
    err = oskar_mem_scale_real(&u, wavenumber);
    if (err) return err;
    err = oskar_mem_scale_real(&v, wavenumber);
    if (err) return err;

    /* Check locations of the data. */
    if (loc_image == OSKAR_LOCATION_CPU)
    {
        err = oskar_mem_init(&t_image, image->type, OSKAR_LOCATION_GPU,
                image->num_elements, 1);

    }

    if (type == OSKAR_DOUBLE)
    {
        /* Call DFT. */
        err = oskar_cuda_dft_c2r_2d_d(num_vis,
                (double*)u.data, (double*)v.data, (double*)t_amp->data,
                num_pixels, d_l, d_m, d_image);
    }
    else if (type == OSKAR_SINGLE)
    {
        /* Call DFT. */
        err = oskar_cuda_dft_c2r_2d_f(num_vis, d_u, d_v, (double*)d_vis,
                num_pixels, d_l, d_m, d_image);
    }

    /* Copy back image to host memory. */
    cudaMemcpy(image, d_image, mem_size_image, cudaMemcpyDeviceToHost);

    for (unsigned i = 0; i < num_pixels; ++i)
    {
        image[i] /= (double)num_vis;
    }

    cleanup:
    /* Free memory. */
    oskar_mem_free(&u);
    oskar_mem_free(&v);
    return err;
}
#endif

#ifdef __cplusplus
}
#endif
