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

#include "imager/oskar_imager_linear_to_stokes.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_linear_to_stokes(const oskar_Mem* in, oskar_Mem** out,
        int* status)
{
    size_t i, num;
    if (*status) return;
    num = oskar_mem_length(in);

    /* Create output array or resize if required. */
    if (!*out)
        *out = oskar_mem_create(oskar_mem_type(in), OSKAR_CPU, num, status);
    else if (oskar_mem_length(*out) < num)
        oskar_mem_realloc(*out, num, status);

    /* Copy or convert if required. */
    if (!oskar_mem_is_matrix(in)) /* Already Stokes I. */
        oskar_mem_copy_contents(*out, in, 0, 0, num, status);
    else
    {
        if (oskar_mem_precision(in) == OSKAR_DOUBLE)
        {
            const double4c* d_ = oskar_mem_double4c_const(in, status);
            double4c* s_ = oskar_mem_double4c(*out, status);
            for (i = 0; i < num; ++i)
            {
                /* I = 0.5 (XX + YY) */
                s_[i].a.x =  0.5 * (d_[i].a.x + d_[i].d.x);
                s_[i].a.y =  0.5 * (d_[i].a.y + d_[i].d.y);
                /* Q = 0.5 (XX - YY) */
                s_[i].b.x =  0.5 * (d_[i].a.x - d_[i].d.x);
                s_[i].b.y =  0.5 * (d_[i].a.y - d_[i].d.y);
                /* U = 0.5 (XY + YX) */
                s_[i].c.x =  0.5 * (d_[i].b.x + d_[i].c.x);
                s_[i].c.y =  0.5 * (d_[i].b.y + d_[i].c.y);
                /* V = -0.5i (XY - YX) */
                s_[i].d.x =  0.5 * (d_[i].b.y - d_[i].c.y);
                s_[i].d.y = -0.5 * (d_[i].b.x - d_[i].c.x);
            }
        }
        else
        {
            const float4c* d_ = oskar_mem_float4c_const(in, status);
            float4c* s_ = oskar_mem_float4c(*out, status);
            for (i = 0; i < num; ++i)
            {
                /* I = 0.5 (XX + YY) */
                s_[i].a.x =  0.5 * (d_[i].a.x + d_[i].d.x);
                s_[i].a.y =  0.5 * (d_[i].a.y + d_[i].d.y);
                /* Q = 0.5 (XX - YY) */
                s_[i].b.x =  0.5 * (d_[i].a.x - d_[i].d.x);
                s_[i].b.y =  0.5 * (d_[i].a.y - d_[i].d.y);
                /* U = 0.5 (XY + YX) */
                s_[i].c.x =  0.5 * (d_[i].b.x + d_[i].c.x);
                s_[i].c.y =  0.5 * (d_[i].b.y + d_[i].c.y);
                /* V = -0.5i (XY - YX) */
                s_[i].d.x =  0.5 * (d_[i].b.y - d_[i].c.y);
                s_[i].d.y = -0.5 * (d_[i].b.x - d_[i].c.x);
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
