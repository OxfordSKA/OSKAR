/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#include <oskar_make_image_dft.h>
#include <oskar_dft_c2r_2d_cuda.h>
#include <oskar_cuda_check_error.h>

#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
extern "C" {
#endif

int oskar_make_image_dft(oskar_Mem* image, const oskar_Mem* uu_metres,
        const oskar_Mem* vv_metres, const oskar_Mem* amp, const oskar_Mem* l,
        const oskar_Mem* m, double frequency_hz)
{
    oskar_Mem u, v, t_l, t_m, t_amp, t_image, *p_image;
    const oskar_Mem *p_l, *p_m, *p_amp;
    double wavenumber;
    int err = 0, type, num_vis, num_pixels;

    /* Check types. */
    type = oskar_mem_type(image);
    if (type != oskar_mem_type(uu_metres) || type != oskar_mem_type(vv_metres) ||
            (type | OSKAR_COMPLEX) != oskar_mem_type(amp) ||
            type != oskar_mem_type(l) || type != oskar_mem_type(m))
        return OSKAR_ERR_TYPE_MISMATCH;

    /* Get dimension sizes. */
    num_vis = (int)oskar_mem_length(amp);
    num_pixels = (int)oskar_mem_length(l);
    if (num_pixels != (int)oskar_mem_length(m))
        return OSKAR_ERR_DIMENSION_MISMATCH;

    /* Initialise all temporary arrays (to zero length). */
    oskar_mem_init(&u, oskar_mem_type(uu_metres), OSKAR_LOCATION_GPU, 0, 1, &err);
    oskar_mem_init(&v, oskar_mem_type(vv_metres), OSKAR_LOCATION_GPU, 0, 1, &err);
    oskar_mem_init(&t_l, oskar_mem_type(l), OSKAR_LOCATION_GPU, 0, 1, &err);
    oskar_mem_init(&t_m, oskar_mem_type(m), OSKAR_LOCATION_GPU, 0, 1, &err);
    oskar_mem_init(&t_amp, oskar_mem_type(amp), OSKAR_LOCATION_GPU, 0, 1, &err);
    oskar_mem_init(&t_image, oskar_mem_type(image), OSKAR_LOCATION_GPU, 0, 1, &err);

    /* Copy the baselines to temporary GPU memory. */
    oskar_mem_copy(&u, uu_metres, &err);
    oskar_mem_copy(&v, vv_metres, &err);

    /* Multiply baselines by the wavenumber. */
    wavenumber = 2.0 * M_PI * frequency_hz / 299792458.0;
    oskar_mem_scale_real(&u, wavenumber, &err);
    oskar_mem_scale_real(&v, wavenumber, &err);

    /* Check location of the image array. */
    p_image = image;
    if (oskar_mem_location(image) == OSKAR_LOCATION_CPU)
    {
        oskar_mem_init(&t_image, oskar_mem_type(image), OSKAR_LOCATION_GPU,
                (int)oskar_mem_length(image), 1, &err);
        p_image = &t_image;
    }

    /* Check location of the l,m arrays. */
    p_l = l;
    p_m = m;
    if (oskar_mem_location(l) == OSKAR_LOCATION_CPU)
    {
        oskar_mem_init(&t_l, oskar_mem_type(l), OSKAR_LOCATION_GPU,
                (int)oskar_mem_length(l), 1, &err);
        oskar_mem_copy(&t_l, l, &err);
        p_l = &t_l;
    }
    if (oskar_mem_location(m) == OSKAR_LOCATION_CPU)
    {
        oskar_mem_init(&t_m, oskar_mem_type(m), OSKAR_LOCATION_GPU,
                (int)oskar_mem_length(m), 1, &err);
        oskar_mem_copy(&t_m, m, &err);
        p_m = &t_m;
    }

    /* Check location of the amplitude array. */
    p_amp = amp;
    if (oskar_mem_location(amp) == OSKAR_LOCATION_CPU)
    {
        oskar_mem_init(&t_amp, oskar_mem_type(amp), OSKAR_LOCATION_GPU,
                (int)oskar_mem_length(amp), 1, &err);
        oskar_mem_copy(&t_amp, amp, &err);
        p_amp = &t_amp;
    }

    /* Check if safe to proceed. */
    if (err) goto cleanup;

    if (type == OSKAR_DOUBLE)
    {
        /* Call DFT. */
        oskar_dft_c2r_2d_cuda_d(num_vis, (double*)(u.data), (double*)(v.data),
                (const double2*)(p_amp->data), num_pixels,
                (const double*)(p_l->data), (const double*)(p_m->data),
                (double*)(p_image->data));
        oskar_cuda_check_error(&err);
    }
    else if (type == OSKAR_SINGLE)
    {
        /* Call DFT. */
        oskar_dft_c2r_2d_cuda_f(num_vis, (float*)(u.data), (float*)(v.data),
                (const float2*)(p_amp->data), num_pixels,
                (const float*)(p_l->data), (const float*)(p_m->data),
                (float*)(p_image->data));
        oskar_cuda_check_error(&err);
    }

    /* Scale image by inverse of number of visibilities. */
    oskar_mem_scale_real(p_image, 1.0 / num_vis, &err);

    /* Copy image back to host memory if required. */
    if (oskar_mem_location(image) == OSKAR_LOCATION_CPU)
        oskar_mem_insert(image, &t_image, 0, &err);

    cleanup:
    /* Free temporary memory. */
    oskar_mem_free(&u, &err);
    oskar_mem_free(&v, &err);
    oskar_mem_free(&t_l, &err);
    oskar_mem_free(&t_m, &err);
    oskar_mem_free(&t_amp, &err);
    oskar_mem_free(&t_image, &err);
    return err;
}

#ifdef __cplusplus
}
#endif
