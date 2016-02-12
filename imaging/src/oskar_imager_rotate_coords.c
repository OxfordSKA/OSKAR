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

#include <oskar_imager_rotate_coords.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_rotate_coords(size_t num_coords, const oskar_Mem* uu_in,
        const oskar_Mem* vv_in, const oskar_Mem* ww_in, const double M[9],
        oskar_Mem* uu, oskar_Mem* vv, oskar_Mem* ww)
{
    register size_t i;
    register double s[3], t[3];
    if (oskar_mem_precision(uu_in) == OSKAR_SINGLE)
    {
        float *uu_o, *vv_o, *ww_o;
        const float *uu_i, *vv_i, *ww_i;
        uu_i = (const float*)oskar_mem_void_const(uu_in);
        vv_i = (const float*)oskar_mem_void_const(vv_in);
        ww_i = (const float*)oskar_mem_void_const(ww_in);
        uu_o = (float*)oskar_mem_void(uu);
        vv_o = (float*)oskar_mem_void(vv);
        ww_o = (float*)oskar_mem_void(ww);
        for (i = 0; i < num_coords; ++i)
        {
            s[0] = uu_i[i]; s[1] = vv_i[i]; s[2] = ww_i[i];
            t[0] = M[0] * s[0] + M[1] * s[1] + M[2] * s[2];
            t[1] = M[3] * s[0] + M[4] * s[1] + M[5] * s[2];
            t[2] = M[6] * s[0] + M[7] * s[1] + M[8] * s[2];
            uu_o[i] = t[0]; vv_o[i] = t[1]; ww_o[i] = t[2];
        }
    }
    else
    {
        double *uu_o, *vv_o, *ww_o;
        const double *uu_i, *vv_i, *ww_i;
        uu_i = (const double*)oskar_mem_void_const(uu_in);
        vv_i = (const double*)oskar_mem_void_const(vv_in);
        ww_i = (const double*)oskar_mem_void_const(ww_in);
        uu_o = (double*)oskar_mem_void(uu);
        vv_o = (double*)oskar_mem_void(vv);
        ww_o = (double*)oskar_mem_void(ww);
        for (i = 0; i < num_coords; ++i)
        {
            s[0] = uu_i[i]; s[1] = vv_i[i]; s[2] = ww_i[i];
            t[0] = M[0] * s[0] + M[1] * s[1] + M[2] * s[2];
            t[1] = M[3] * s[0] + M[4] * s[1] + M[5] * s[2];
            t[2] = M[6] * s[0] + M[7] * s[1] + M[8] * s[2];
            uu_o[i] = t[0]; vv_o[i] = t[1]; ww_o[i] = t[2];
        }
    }
}

#ifdef __cplusplus
}
#endif
