/*
 * Copyright (c) 2016-2019, The University of Oxford
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

#include "imager/define_imager_generate_w_phase_screen.h"
#include "imager/private_imager_generate_w_phase_screen.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_IMAGER_GENERATE_W_PHASE_SCREEN(imager_generate_w_phase_screen_float, float)
OSKAR_IMAGER_GENERATE_W_PHASE_SCREEN(imager_generate_w_phase_screen_double, double)

void oskar_imager_generate_w_phase_screen(int iw, int conv_size, int inner,
        double sampling, double w_scale, const oskar_Mem* taper_func,
        oskar_Mem* screen, int* status)
{
    const int location = oskar_mem_location(screen);
    const int type = oskar_mem_precision(screen);
    const int inner_half = inner / 2;
    const double f = (2.0 * M_PI * iw * iw) / w_scale;
    const float f_f = (float)f;
    const float sampling_f = (float)sampling;
    oskar_mem_clear_contents(screen, status);
    if (*status) return;
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
            imager_generate_w_phase_screen_float(
                    conv_size, inner_half, sampling_f, f_f,
                    oskar_mem_float_const(taper_func, status),
                    oskar_mem_float(screen, status));
        else if (type == OSKAR_DOUBLE)
            imager_generate_w_phase_screen_double(
                    conv_size, inner_half, sampling, f,
                    oskar_mem_double_const(taper_func, status),
                    oskar_mem_double(screen, status));
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else
    {
        size_t local_size[] = {16, 16, 1}, global_size[] = {1, 1, 1};
        const int is_dbl = (type == OSKAR_DOUBLE);
        const char* k = 0;
        if (type == OSKAR_SINGLE)
            k = "imager_generate_w_phase_screen_float";
        else if (type == OSKAR_DOUBLE)
            k = "imager_generate_w_phase_screen_double";
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        oskar_device_check_local_size(location, 1, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) inner, local_size[0]);
        global_size[1] = oskar_device_global_size(
                (size_t) inner, local_size[1]);
        const oskar_Arg args[] = {
                {INT_SZ, &conv_size},
                {INT_SZ, &inner_half},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&sampling : (const void*)&sampling_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&f : (const void*)&f_f},
                {PTR_SZ, oskar_mem_buffer_const(taper_func)},
                {PTR_SZ, oskar_mem_buffer(screen)}
        };
        oskar_device_launch_kernel(k, location, 2, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
