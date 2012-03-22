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


#include "imaging/oskar_get_image_stokes.h"
#include "utility/oskar_mem_type_check.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_assign.h"
#include "utility/oskar_vector_types.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_get_image_stokes(oskar_Mem* stokes, const oskar_Visibilities* vis,
        const oskar_SettingsImage* settings)
{
    int num_vis_pols;
    int pol;
    int num_vis_amps;
    int type;
    int i;
    int location;

    /* Local variables */
    num_vis_pols = oskar_mem_is_matrix(vis->amplitude.type) ? 4 : 1;
    pol = settings->polarisation;
    num_vis_amps = vis->num_baselines * vis->num_times * vis->num_channels;
    type = oskar_mem_is_double(vis->amplitude.type) ? OSKAR_DOUBLE : OSKAR_SINGLE;
    location = OSKAR_LOCATION_CPU;

    /* If the input data is polarised and a single stokes polarisation type
     * is selected */
    if (num_vis_pols == 4 && (pol == OSKAR_IMAGE_TYPE_STOKES_I ||
            pol == OSKAR_IMAGE_TYPE_STOKES_Q ||
            pol == OSKAR_IMAGE_TYPE_STOKES_U ||
            pol == OSKAR_IMAGE_TYPE_STOKES_V))
    {
        /* I = 0.5 (XX + YY) */
        switch (pol)
        {
            case OSKAR_IMAGE_TYPE_STOKES_I:
            {
                oskar_mem_init(stokes, type | OSKAR_COMPLEX, location,
                        num_vis_amps, OSKAR_TRUE);
                for (i = 0; i < num_vis_amps; ++i)
                {
                    if (type == OSKAR_DOUBLE)
                    {
                        double4c* d_ = (double4c*)vis->amplitude.data;
                        double2* s_ = (double2*)stokes->data;
                        s_[i].x = 0.5 * (d_[i].a.x + d_[i].d.x);
                        s_[i].y = 0.5 * (d_[i].a.y + d_[i].d.y);
                    }
                    else
                    {
                        float4c* d_ = (float4c*)vis->amplitude.data;
                        float2* s_ = (float2*)stokes->data;
                        s_[i].x = 0.5 * (d_[i].a.x + d_[i].d.x);
                        s_[i].y = 0.5 * (d_[i].a.y + d_[i].d.y);
                    }
                }
                break;
            }
            case OSKAR_IMAGE_TYPE_STOKES_Q:
            {
                oskar_mem_init(stokes, type | OSKAR_COMPLEX, location,
                        num_vis_amps, OSKAR_TRUE);
                for (i = 0; i < num_vis_amps; ++i)
                {
                    if (type == OSKAR_DOUBLE)
                    {
                        double4c* d_ = (double4c*)vis->amplitude.data;
                        double2* s_ = (double2*)stokes->data;
                        s_[i].x = 0.5 * (d_[i].a.x - d_[i].d.x);
                        s_[i].y = 0.5 * (d_[i].a.y - d_[i].d.y);
                    }
                    else
                    {
                        float4c* d_ = (float4c*)vis->amplitude.data;
                        float2* s_ = (float2*)stokes->data;
                        s_[i].x = 0.5 * (d_[i].a.x - d_[i].d.x);
                        s_[i].y = 0.5 * (d_[i].a.y - d_[i].d.y);
                    }
                }
                break;
            }
            case OSKAR_IMAGE_TYPE_STOKES_U:
            {
                oskar_mem_init(stokes, type | OSKAR_COMPLEX, location,
                        num_vis_amps, OSKAR_TRUE);
                for (i = 0; i < num_vis_amps; ++i)
                {
                    if (type == OSKAR_DOUBLE)
                    {
                        double4c* d_ = (double4c*)vis->amplitude.data;
                        double2* s_ = (double2*)stokes->data;
                        s_[i].x = 0.5 * (d_[i].b.x + d_[i].c.x);
                        s_[i].y = 0.5 * (d_[i].b.y + d_[i].c.y);
                    }
                    else
                    {
                        float4c* d_ = (float4c*)vis->amplitude.data;
                        float2* s_ = (float2*)stokes->data;
                        s_[i].x = 0.5 * (d_[i].b.x + d_[i].c.x);
                        s_[i].y = 0.5 * (d_[i].b.y + d_[i].c.y);
                    }
                }
                break;
            }
            case OSKAR_IMAGE_TYPE_STOKES_V:
            {
                oskar_mem_init(stokes, type | OSKAR_COMPLEX, location,
                        num_vis_amps, OSKAR_TRUE);
                for (i = 0; i < num_vis_amps; ++i)
                {
                    if (type == OSKAR_DOUBLE)
                    {
                        double4c* d_ = (double4c*)vis->amplitude.data;
                        double2* s_ = (double2*)stokes->data;
                        s_[i].x =  0.5 * (d_[i].b.y - d_[i].c.y);
                        s_[i].y = -0.5 * (d_[i].b.x - d_[i].c.x);
                    }
                    else
                    {
                        float4c* d_ = (float4c*)vis->amplitude.data;
                        float2* s_ = (float2*)stokes->data;
                        s_[i].x =  0.5 * (d_[i].b.y - d_[i].c.y);
                        s_[i].y = -0.5 * (d_[i].b.x - d_[i].c.x);
                    }
                }
                break;
            }
            default:
            {
                return OSKAR_ERR_BAD_DATA_TYPE;
            }
        }; /* switch (pol_type) */
    }
    else if (num_vis_pols == 4 && pol == OSKAR_IMAGE_TYPE_STOKES)
    {
        oskar_mem_init(stokes, vis->amplitude.type, location, num_vis_amps,
                OSKAR_TRUE);
        for (i = 0; i < num_vis_amps; ++i)
        {
            if (type == OSKAR_DOUBLE)
            {
                double4c* d_ = (double4c*)vis->amplitude.data;
                double4c* s_ = (double4c*)stokes->data;
                /* I = 0.5 (XX + YY) */
                s_[i].a.x =  0.5 * (d_[i].a.x + d_[i].d.x);
                s_[i].a.y =  0.5 * (d_[i].a.y + d_[i].d.y);
                /* Q = 0.5 (XX - YY)*/
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
                float4c* d_ = (float4c*)vis->amplitude.data;
                float4c* s_ = (float4c*)stokes->data;
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
    else if (num_vis_pols == 1)
    {
        /* TODO: check this branch very carefully! */
        if (pol != OSKAR_IMAGE_TYPE_STOKES_I) return OSKAR_ERR_UNKNOWN;
        oskar_mem_init(stokes, vis->amplitude.type, location, num_vis_amps,
                OSKAR_FALSE);
        oskar_mem_assign(stokes, &vis->amplitude);
    }

    return OSKAR_SUCCESS;
}


#ifdef __cplusplus
}
#endif
