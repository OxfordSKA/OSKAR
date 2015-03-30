/*
 * Copyright (c) 2011-2015, The University of Oxford
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

#include <oskar_cmath.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_make_image_dft(oskar_Mem* image, const oskar_Mem* uu_metres,
        const oskar_Mem* vv_metres, const oskar_Mem* amp, const oskar_Mem* l,
        const oskar_Mem* m, double frequency_hz, int* status)
{
    oskar_Mem *u, *v, *t_l = 0, *t_m = 0, *t_amp = 0, *t_image = 0, *p_image;
    const oskar_Mem *p_l, *p_m, *p_amp;
    double wavenumber;
    int type, num_vis, num_pixels;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check types. */
    type = oskar_mem_type(image);
    if (type != oskar_mem_type(uu_metres) ||
            type != oskar_mem_type(vv_metres) ||
            (type | OSKAR_COMPLEX) != oskar_mem_type(amp) ||
            type != oskar_mem_type(l) || type != oskar_mem_type(m))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }

    /* Get dimension sizes. */
    num_vis = (int)oskar_mem_length(amp);
    num_pixels = (int)oskar_mem_length(l);
    if (num_pixels != (int)oskar_mem_length(m))
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Copy the baselines to temporary GPU memory and compute the wavenumber. */
    u = oskar_mem_create_copy(uu_metres, OSKAR_GPU, status);
    v = oskar_mem_create_copy(vv_metres, OSKAR_GPU, status);
    wavenumber = 2.0 * M_PI * frequency_hz / 299792458.0;

    /* Check location of the image array. */
    p_image = image;
    if (oskar_mem_location(image) == OSKAR_CPU)
    {
        t_image = oskar_mem_create(oskar_mem_type(image), OSKAR_GPU,
                oskar_mem_length(image), status);
        p_image = t_image;
    }

    /* Check location of the l,m arrays. */
    p_l = l;
    p_m = m;
    if (oskar_mem_location(l) == OSKAR_CPU)
    {
        t_l = oskar_mem_create_copy(l, OSKAR_GPU, status);
        p_l = t_l;
    }
    if (oskar_mem_location(m) == OSKAR_CPU)
    {
        t_m = oskar_mem_create_copy(m, OSKAR_GPU, status);
        p_m = t_m;
    }

    /* Check location of the amplitude array. */
    p_amp = amp;
    if (oskar_mem_location(amp) == OSKAR_CPU)
    {
        t_amp = oskar_mem_create_copy(amp, OSKAR_GPU, status);
        p_amp = t_amp;
    }

    /* Check if safe to proceed. */
    if (!*status)
    {
        if (type == OSKAR_DOUBLE)
        {
            /* Call DFT. */
            oskar_dft_c2r_2d_cuda_d(num_vis, wavenumber,
                    oskar_mem_double_const(u, status),
                    oskar_mem_double_const(v, status),
                    oskar_mem_double2_const(p_amp, status), num_pixels,
                    oskar_mem_double_const(p_l, status),
                    oskar_mem_double_const(p_m, status),
                    oskar_mem_double(p_image, status));
            oskar_cuda_check_error(status);
        }
        else if (type == OSKAR_SINGLE)
        {
            /* Call DFT. */
            oskar_dft_c2r_2d_cuda_f(num_vis, wavenumber,
                    oskar_mem_float_const(u, status),
                    oskar_mem_float_const(v, status),
                    oskar_mem_float2_const(p_amp, status), num_pixels,
                    oskar_mem_float_const(p_l, status),
                    oskar_mem_float_const(p_m, status),
                    oskar_mem_float(p_image, status));
            oskar_cuda_check_error(status);
        }

        /* Scale image by inverse of number of visibilities. */
        oskar_mem_scale_real(p_image, 1.0 / num_vis, status);

        /* Copy image back to host memory if required. */
        if (oskar_mem_location(image) == OSKAR_CPU)
            oskar_mem_copy_contents(image, t_image, 0, 0,
                    oskar_mem_length(t_image), status);
    }

    /* Free temporary memory. */
    oskar_mem_free(u, status);
    oskar_mem_free(v, status);
    oskar_mem_free(t_l, status);
    oskar_mem_free(t_m, status);
    oskar_mem_free(t_amp, status);
    oskar_mem_free(t_image, status);
}

#ifdef __cplusplus
}
#endif
