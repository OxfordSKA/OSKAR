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

#include <cufft.h>
#include <private_imager.h>
#include <private_imager_algorithm_free_fft.h>
#include <private_imager_algorithm_init_fft.h>

#include <oskar_cmath.h>
#include <oskar_convert_fov_to_cellsize.h>
#include <oskar_fftpack_cfft.h>
#include <oskar_fftpack_cfft_f.h>
#include <oskar_grid_functions_spheroidal.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_algorithm_init_fft(oskar_Imager* h, int* status)
{
    oskar_imager_algorithm_free_fft(h, status);
    if (*status) return;

    /* Calculate cellsize. */
    h->cellsize_rad = oskar_convert_fov_to_cellsize(h->fov_deg * M_PI/180,
            h->size);

    /* Generate the convolution function. */
    h->conv_func = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU,
            h->oversample * (h->support + 1), status);
    h->corr_func = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU,
            h->size, status);
    oskar_grid_convolution_function_spheroidal(h->support, h->oversample,
            oskar_mem_double(h->conv_func, status));
    oskar_grid_correction_function_spheroidal(h->size,
            oskar_mem_double(h->corr_func, status));

    /* Set up the FFT. */
    if (h->fft_on_gpu)
    {
        /* Generate FFT plan. */
        if (h->imager_prec == OSKAR_DOUBLE)
            cufftPlan2d(&h->cufft_plan_imager, h->size, h->size, CUFFT_Z2Z);
        else
            cufftPlan2d(&h->cufft_plan_imager, h->size, h->size, CUFFT_C2C);
    }
    else
    {
        /* Initialise workspaces for CPU FFT algorithm. */
        int len_save = 4 * h->size +
                2 * (int)(log((double)h->size) / log(2.0)) + 8;
        h->wsave = oskar_mem_create(h->imager_prec, OSKAR_CPU,
                len_save, status);
        h->work = oskar_mem_create(h->imager_prec, OSKAR_CPU,
                2 * h->size * h->size, status);
        if (h->imager_prec == OSKAR_SINGLE)
            oskar_fftpack_cfft2i_f(h->size, h->size,
                    oskar_mem_float(h->wsave, status));
        else
            oskar_fftpack_cfft2i(h->size, h->size,
                    oskar_mem_double(h->wsave, status));
    }
}

#ifdef __cplusplus
}
#endif
