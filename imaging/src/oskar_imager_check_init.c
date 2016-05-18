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
#include <oskar_imager.h>

#include <oskar_imager_check_init.h>
#include <private_imager_init_dft.h>
#include <private_imager_init_fft.h>
#include <private_imager_init_wproj.h>

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_check_init(oskar_Imager* h, int* status)
{
    switch (h->algorithm)
    {
    case OSKAR_ALGORITHM_DFT_2D:
    case OSKAR_ALGORITHM_DFT_3D:
    {
        if (!h->l)
            oskar_imager_init_dft(h, status);
        break;
    }
    case OSKAR_ALGORITHM_FFT:
    {
        if (!h->conv_func)
            oskar_imager_init_fft(h, status);
        break;
    }
    case OSKAR_ALGORITHM_WPROJ:
    {
        if (!h->w_kernels)
            oskar_imager_init_wproj(h, status);
        break;
    }
    default:
        *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    }
}

#ifdef __cplusplus
}
#endif
