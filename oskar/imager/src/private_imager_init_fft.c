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

#include "imager/private_imager.h"

#include "imager/private_imager_init_fft.h"
#include "imager/oskar_grid_functions_spheroidal.h"
#include "imager/oskar_grid_functions_pillbox.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_init_fft(oskar_Imager* h, int* status)
{
    oskar_Mem* tmp = 0;
    if (*status) return;

    /* Generate the convolution function. */
    tmp = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU,
            h->oversample * (h->support + 1), status);
    switch (h->kernel_type)
    {
    case 'S':
        oskar_grid_convolution_function_spheroidal(h->support, h->oversample,
                oskar_mem_double(tmp, status));
        break;
    case 'P':
        oskar_grid_convolution_function_pillbox(h->support, h->oversample,
                oskar_mem_double(tmp, status));
        break;
    default:
        *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
        break;
    }

    /* Save the convolution function with appropriate numerical precision. */
    oskar_mem_free(h->conv_func, status);
    h->conv_func = oskar_mem_convert_precision(tmp, h->imager_prec, status);
    oskar_mem_free(tmp, status);
}

#ifdef __cplusplus
}
#endif
