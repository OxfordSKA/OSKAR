/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include <oskar_image.h>
#include <oskar_get_image_stokes.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Mem* oskar_get_image_stokes(const oskar_Vis* vis,
        const oskar_SettingsImage* settings, int* status)
{
    int num_vis_pols, num_vis_amps, im_type, type, i, location;
    const oskar_Mem* vis_amp;
    oskar_Mem* stokes = 0;

    if (*status) return 0;

    /* Local variables */
    im_type = settings->image_type;
    if (im_type == OSKAR_IMAGE_TYPE_PSF) return 0;
    vis_amp = oskar_vis_amplitude_const(vis);
    num_vis_pols = oskar_mem_is_matrix(vis_amp) ? 4 : 1;
    num_vis_amps = oskar_vis_num_baselines(vis) * oskar_vis_num_times(vis) *
            oskar_vis_num_channels(vis);
    type = oskar_mem_precision(vis_amp);
    location = OSKAR_CPU;

    /* CASE1: Polarised vis amplitudes, single Stokes output Image */
    if (num_vis_pols == 4 && (im_type == OSKAR_IMAGE_TYPE_STOKES_I ||
            im_type == OSKAR_IMAGE_TYPE_STOKES_Q ||
            im_type == OSKAR_IMAGE_TYPE_STOKES_U ||
            im_type == OSKAR_IMAGE_TYPE_STOKES_V))
    {
        /* I = 0.5 (XX + YY) */
        switch (im_type)
        {
            case OSKAR_IMAGE_TYPE_STOKES_I:
            {
                stokes = oskar_mem_create(type | OSKAR_COMPLEX, location,
                        num_vis_amps, status);
                if (*status) return 0;
                for (i = 0; i < num_vis_amps; ++i)
                {
                    if (type == OSKAR_DOUBLE)
                    {
                        const double4c* d_ = oskar_mem_double4c_const(vis_amp, status);
                        double2* s_ = oskar_mem_double2(stokes, status);
                        s_[i].x = 0.5 * (d_[i].a.x + d_[i].d.x);
                        s_[i].y = 0.5 * (d_[i].a.y + d_[i].d.y);
                    }
                    else
                    {
                        const float4c* d_ = oskar_mem_float4c_const(vis_amp, status);
                        float2* s_ = oskar_mem_float2(stokes, status);
                        s_[i].x = 0.5 * (d_[i].a.x + d_[i].d.x);
                        s_[i].y = 0.5 * (d_[i].a.y + d_[i].d.y);
                    }
                }
                break;
            }
            case OSKAR_IMAGE_TYPE_STOKES_Q:
            {
                stokes = oskar_mem_create(type | OSKAR_COMPLEX, location,
                        num_vis_amps, status);
                if (*status) return 0;
                for (i = 0; i < num_vis_amps; ++i)
                {
                    if (type == OSKAR_DOUBLE)
                    {
                        const double4c* d_ = oskar_mem_double4c_const(vis_amp, status);
                        double2* s_ = oskar_mem_double2(stokes, status);
                        s_[i].x = 0.5 * (d_[i].a.x - d_[i].d.x);
                        s_[i].y = 0.5 * (d_[i].a.y - d_[i].d.y);
                    }
                    else
                    {
                        const float4c* d_ = oskar_mem_float4c_const(vis_amp, status);
                        float2* s_ = oskar_mem_float2(stokes, status);
                        s_[i].x = 0.5 * (d_[i].a.x - d_[i].d.x);
                        s_[i].y = 0.5 * (d_[i].a.y - d_[i].d.y);
                    }
                }
                break;
            }
            case OSKAR_IMAGE_TYPE_STOKES_U:
            {
                stokes = oskar_mem_create(type | OSKAR_COMPLEX, location,
                        num_vis_amps, status);
                if (*status) return 0;
                for (i = 0; i < num_vis_amps; ++i)
                {
                    if (type == OSKAR_DOUBLE)
                    {
                        const double4c* d_ = oskar_mem_double4c_const(vis_amp, status);
                        double2* s_ = oskar_mem_double2(stokes, status);
                        s_[i].x = 0.5 * (d_[i].b.x + d_[i].c.x);
                        s_[i].y = 0.5 * (d_[i].b.y + d_[i].c.y);
                    }
                    else
                    {
                        const float4c* d_ = oskar_mem_float4c_const(vis_amp, status);
                        float2* s_ = oskar_mem_float2(stokes, status);
                        s_[i].x = 0.5 * (d_[i].b.x + d_[i].c.x);
                        s_[i].y = 0.5 * (d_[i].b.y + d_[i].c.y);
                    }
                }
                break;
            }
            case OSKAR_IMAGE_TYPE_STOKES_V:
            {
                stokes = oskar_mem_create(type | OSKAR_COMPLEX, location,
                        num_vis_amps, status);
                if (*status) return 0;
                for (i = 0; i < num_vis_amps; ++i)
                {
                    if (type == OSKAR_DOUBLE)
                    {
                        const double4c* d_ = oskar_mem_double4c_const(vis_amp, status);
                        double2* s_ = oskar_mem_double2(stokes, status);
                        s_[i].x =  0.5 * (d_[i].b.y - d_[i].c.y);
                        s_[i].y = -0.5 * (d_[i].b.x - d_[i].c.x);
                    }
                    else
                    {
                        const float4c* d_ = oskar_mem_float4c_const(vis_amp, status);
                        float2* s_ = oskar_mem_float2(stokes, status);
                        s_[i].x =  0.5 * (d_[i].b.y - d_[i].c.y);
                        s_[i].y = -0.5 * (d_[i].b.x - d_[i].c.x);
                    }
                }
                break;
            }
            default:
            {
                *status = OSKAR_ERR_BAD_DATA_TYPE;
                return 0;
            }
        }; /* switch (pol_type) */
    }
    /* CASE2: Polarised vis amplitudes, image of all 4 stokes parameters */
    else if (num_vis_pols == 4 && im_type == OSKAR_IMAGE_TYPE_STOKES)
    {
        stokes = oskar_mem_create(oskar_mem_type(vis_amp), location, num_vis_amps,
                status);
        if (*status) return 0;
        for (i = 0; i < num_vis_amps; ++i)
        {
            if (type == OSKAR_DOUBLE)
            {
                const double4c* d_ = oskar_mem_double4c_const(vis_amp, status);
                double4c* s_ = oskar_mem_double4c(stokes, status);
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
            else
            {
                const float4c* d_ = oskar_mem_float4c_const(vis_amp, status);
                float4c* s_ = oskar_mem_float4c(stokes, status);
                /* I */
                s_[i].a.x =  0.5 * (d_[i].a.x + d_[i].d.x);
                s_[i].a.y =  0.5 * (d_[i].a.y + d_[i].d.y);
                /* Q */
                s_[i].b.x =  0.5 * (d_[i].a.x - d_[i].d.x);
                s_[i].b.y =  0.5 * (d_[i].a.y - d_[i].d.y);
                /* U */
                s_[i].c.x =  0.5 * (d_[i].b.x + d_[i].c.x);
                s_[i].c.y =  0.5 * (d_[i].b.y + d_[i].c.y);
                /* V */
                s_[i].d.x =  0.5 * (d_[i].b.y - d_[i].c.y);
                s_[i].d.y = -0.5 * (d_[i].b.x - d_[i].c.x);
            }
        }
    }
    /* CASE3: Scalar (Stokes-I) vis amplitudes, require the output image
     * to also be Stokes-I */
    else if (num_vis_pols == 1)
    {
        /* TODO better logic for scalar mode vis? */
        if (im_type != OSKAR_IMAGE_TYPE_STOKES_I)
        {
            *status = OSKAR_ERR_SETTINGS_IMAGE;
            return 0;
        }
        stokes = oskar_mem_create_alias(vis_amp, 0, oskar_mem_length(vis_amp), status);
    }

    return stokes;
}

#ifdef __cplusplus
}
#endif
