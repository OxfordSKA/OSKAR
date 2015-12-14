/*
 * Copyright (c) 2012-2014, The University of Oxford
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
#include <oskar_get_image_vis_amps.h>
#include <oskar_evaluate_image_ranges.h>
#include <oskar_vector_types.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_get_image_vis_amps(oskar_Mem* amps, const oskar_Vis* vis,
        const oskar_Mem* stokes, const oskar_SettingsImage* settings,
        int vis_channel, int vis_time, int p)
{
    const oskar_Mem* v_ = oskar_vis_amplitude_const(vis);
    int num_channels = oskar_vis_num_channels(vis);
    int num_times = oskar_vis_num_times(vis);
    int num_baselines = oskar_vis_num_baselines(vis);

    int err = 0;
    int pol = settings->image_type;
    int num_pols = (pol ==  OSKAR_IMAGE_TYPE_STOKES ||
            pol == OSKAR_IMAGE_TYPE_POL_LINEAR) ? 4 : 1;

    // Data ranges for frequency and time synthesis.
    int vis_time_range[2], vis_chan_range[2];
    oskar_evaluate_image_data_range(vis_time_range, settings->time_range,
            num_times, &err);
    oskar_evaluate_image_data_range(vis_chan_range, settings->channel_range,
            num_channels, &err);
    if (err) return err;

    /* ================================================================= */
    if (oskar_mem_precision(v_) == OSKAR_DOUBLE)
    {
        double2* a_ = oskar_mem_double2(amps, &err);

        /* ----------------------------------- TIME SNAPSHOTS, FREQ SNAPSHOTS */
        if (settings->time_snapshots && settings->channel_snapshots)
        {
            int idx = (vis_channel * num_times + vis_time) * num_baselines;
            if (num_pols == 4)
            {
                const double4c* data;
                if (pol == OSKAR_IMAGE_TYPE_POL_LINEAR)
                    data = oskar_mem_double4c_const(v_, &err);
                else /* pol == OSKAR_IMAGE_TYPE_STOKES */
                    data = oskar_mem_double4c_const(stokes, &err);

                for (int b = 0; b < num_baselines; ++b)
                {
                    if (p == 0)
                    {
                        a_[b].x = data[idx+b].a.x; a_[b].y= data[idx+b].a.y;
                    }
                    else if (p == 1)
                    {
                        a_[b].x=data[idx+b].b.x; a_[b].y=data[idx+b].b.y;
                    }
                    else if (p == 2)
                    {
                        a_[b].x=data[idx+b].c.x; a_[b].y=data[idx+b].c.y;
                    }
                    else if (p == 3)
                    {
                        a_[b].x=data[idx+b].d.x; a_[b].y=data[idx+b].d.y;
                    }
                }
            }
            else if (num_pols == 1)
            {
                if (pol == OSKAR_IMAGE_TYPE_STOKES_I ||
                        pol == OSKAR_IMAGE_TYPE_STOKES_Q ||
                        pol == OSKAR_IMAGE_TYPE_STOKES_U ||
                        pol == OSKAR_IMAGE_TYPE_STOKES_V)
                {
                    const double2* data = oskar_mem_double2_const(stokes, &err);
                    for (int b = 0; b < num_baselines; ++b)
                    {
                        a_[b].x = data[idx+b].x; a_[b].y = data[idx+b].y;
                    }
                }
                else
                {
                    const double4c* data = oskar_mem_double4c_const(v_, &err);
                    for (int b = 0; b < num_baselines; ++b)
                    {
                        if (pol == OSKAR_IMAGE_TYPE_POL_XX)
                        {
                            a_[b].x=data[idx+b].a.x; a_[b].y=data[idx+b].a.y;
                        }
                        else if (pol == OSKAR_IMAGE_TYPE_POL_XY)
                        {
                            a_[b].x=data[idx+b].b.x; a_[b].y=data[idx+b].b.y;
                        }
                        else if (pol == OSKAR_IMAGE_TYPE_POL_YX)
                        {
                            a_[b].x=data[idx+b].c.x; a_[b].y=data[idx+b].c.y;
                        }
                        else if (pol == OSKAR_IMAGE_TYPE_POL_YY)
                        {
                            a_[b].x=data[idx+b].d.x; a_[b].y=data[idx+b].d.y;
                        }
                    }
                }
            } /* num_pols == 1 */
        }



        /* ---------------------------------- TIME SNAPSHOTS, FREQ SYNTHESIS */
        else if (settings->time_snapshots && !settings->channel_snapshots)
        {
            if (num_pols == 4)
            {
                const double4c* data;
                if (pol == OSKAR_IMAGE_TYPE_POL_LINEAR)
                    data = oskar_mem_double4c_const(v_, &err);
                else /* pol == OSKAR_IMAGE_TYPE_STOKES */
                    data = oskar_mem_double4c_const(stokes, &err);
                for (int i = 0, c = vis_chan_range[0]; c <= vis_chan_range[1]; ++c)
                {
                    int idx = (c*num_times+vis_time)*num_baselines;
                    for (int b = 0; b < num_baselines;++b, ++i)
                    {
                        if (p == 0)
                        {
                            a_[i].x=data[idx+b].a.x; a_[i].y=data[idx+b].a.y;
                        }
                        else if (p == 1)
                        {
                            a_[i].x=data[idx+b].b.x; a_[i].y=data[idx+b].b.y;
                        }
                        else if (p == 2)
                        {
                            a_[i].x=data[idx+b].c.x; a_[i].y=data[idx+b].c.y;
                        }
                        else if (p == 3)
                        {
                            a_[i].x=data[idx+b].d.x; a_[i].y=data[idx+b].d.y;
                        }
                    }
                }
            }
            else /* (num_pols == 1) */
            {
                if (pol == OSKAR_IMAGE_TYPE_STOKES_I ||
                        pol == OSKAR_IMAGE_TYPE_STOKES_Q ||
                        pol == OSKAR_IMAGE_TYPE_STOKES_U ||
                        pol == OSKAR_IMAGE_TYPE_STOKES_V)
                {
                    const double2* data = oskar_mem_double2_const(stokes, &err);
                    for (int i = 0, c = vis_chan_range[0]; c <= vis_chan_range[1]; ++c)
                    {
                        int idx = (c*num_times+vis_time)*num_baselines;
                        for (int b = 0; b < num_baselines;++b, ++i)
                        {
                            a_[i].x=data[idx+b].x; a_[i].y=data[idx+b].y;
                        }
                    }
                }
                else
                {
                    const double4c* data = oskar_mem_double4c_const(v_, &err);
                    for (int i = 0, c = vis_chan_range[0]; c <= vis_chan_range[1]; ++c)
                    {
                        int idx = (c*num_times+vis_time)*num_baselines;
                        for (int b = 0; b < num_baselines;++b, ++i)
                        {
                            if (pol == OSKAR_IMAGE_TYPE_POL_XX)
                            {
                                a_[i].x=data[idx+b].a.x; a_[i].y=data[idx+b].a.y;
                            }
                            else if (pol == OSKAR_IMAGE_TYPE_POL_XY)
                            {
                                a_[i].x=data[idx+b].b.x; a_[i].y=data[idx+b].b.y;
                            }
                            else if (pol == OSKAR_IMAGE_TYPE_POL_YX)
                            {
                                a_[i].x=data[idx+b].c.x; a_[i].y=data[idx+b].c.y;
                            }
                            else if (pol == OSKAR_IMAGE_TYPE_POL_YY)
                            {
                                a_[i].x=data[idx+b].d.x; a_[i].y=data[idx+b].d.y;
                            }
                        }
                    }
                }
            } /* num_pols == 1 */
        }

        /* ---------------------------------- TIME SYNTHESIS, FREQ SNAPSHOTS */
        else if (!settings->time_snapshots && settings->channel_snapshots)
        {
            if (num_pols == 4)
            {
                const double4c* data;
                if (pol == OSKAR_IMAGE_TYPE_POL_LINEAR)
                    data = oskar_mem_double4c_const(v_, &err);
                else /* pol == OSKAR_IMAGE_TYPE_STOKES */
                    data = oskar_mem_double4c_const(stokes, &err);
                for (int i = 0, t = vis_time_range[0]; t <= vis_time_range[1]; ++t)
                {
                    int idx = (vis_channel*num_times+t)*num_baselines;
                    for (int b = 0; b < num_baselines;++b, ++i)
                    {
                        if (p == 0)
                        {
                            a_[i].x=data[idx+b].a.x; a_[i].y=data[idx+b].a.y;
                        }
                        else if (p == 1)
                        {
                            a_[i].x=data[idx+b].b.x; a_[i].y=data[idx+b].b.y;
                        }
                        else if (p == 2)
                        {
                            a_[i].x=data[idx+b].c.x; a_[i].y=data[idx+b].c.y;
                        }
                        else if (p == 3)
                        {
                            a_[i].x=data[idx+b].d.x; a_[i].y=data[idx+b].d.y;
                        }
                    }
                }
            }
            else /* (num_pols == 1) */
            {
                if (pol == OSKAR_IMAGE_TYPE_STOKES_I ||
                        pol == OSKAR_IMAGE_TYPE_STOKES_Q ||
                        pol == OSKAR_IMAGE_TYPE_STOKES_U ||
                        pol == OSKAR_IMAGE_TYPE_STOKES_V)
                {
                    const double2* data = oskar_mem_double2_const(stokes, &err);
                    for (int i = 0, t = vis_time_range[0]; t <= vis_time_range[1]; ++t)
                    {
                        int idx = (vis_channel*num_times+t)*num_baselines;
                        for (int b = 0; b < num_baselines;++b, ++i)
                        {
                            a_[i].x=data[idx+b].x; a_[i].y=data[idx+b].y;
                        }
                    }
                }
                else
                {
                    const double4c* data = oskar_mem_double4c_const(v_, &err);
                    for (int i = 0, t = vis_time_range[0]; t <= vis_time_range[1]; ++t)
                    {
                        int idx = (vis_channel*num_times+t)*num_baselines;
                        for (int b = 0; b < num_baselines;++b, ++i)
                        {
                            if (pol == OSKAR_IMAGE_TYPE_POL_XX)
                            {
                                a_[i].x=data[idx+b].a.x; a_[i].y=data[idx+b].a.y;
                            }
                            else if (pol == OSKAR_IMAGE_TYPE_POL_XY)
                            {
                                a_[i].x=data[idx+b].b.x; a_[i].y=data[idx+b].b.y;
                            }
                            else if (pol == OSKAR_IMAGE_TYPE_POL_YX)
                            {
                                a_[i].x=data[idx+b].c.x; a_[i].y=data[idx+b].c.y;
                            }
                            else if (pol == OSKAR_IMAGE_TYPE_POL_YY)
                            {
                                a_[i].x=data[idx+b].d.x; a_[i].y=data[idx+b].d.y;
                            }
                        }
                    }
                }
            } /* num_pols == 1 */
        }

        /* ----------------------------------- TIME SYNTHESIS, FREQ SYNTHESIS */
        else
        {
            if (num_pols == 4)
            {
                const double4c* data;
                if (pol == OSKAR_IMAGE_TYPE_POL_LINEAR)
                    data = oskar_mem_double4c_const(v_, &err);
                else /* pol == OSKAR_IMAGE_TYPE_STOKES */
                    data = oskar_mem_double4c_const(stokes, &err);
                for (int i = 0, c = vis_chan_range[0]; c <= vis_chan_range[1]; ++c)
                {
                    for (int t = vis_time_range[0]; t <=vis_time_range[1]; ++t)
                    {
                        int idx = (c*num_times+t)*num_baselines;
                        for (int b = 0; b < num_baselines;++b, ++i)
                        {
                            if (p == 0)
                            {
                                a_[i].x=data[idx+b].a.x; a_[i].y=data[idx+b].a.y;
                            }
                            else if (p == 1)
                            {
                                a_[i].x=data[idx+b].b.x; a_[i].y=data[idx+b].b.y;
                            }
                            else if (p == 2)
                            {
                                a_[i].x=data[idx+b].c.x; a_[i].y=data[idx+b].c.y;
                            }
                            else if (p == 3)
                            {
                                a_[i].x=data[idx+b].d.x; a_[i].y=data[idx+b].d.y;
                            }
                        }
                    }
                }
            }
            else
            {
                if (pol == OSKAR_IMAGE_TYPE_STOKES_I ||
                        pol == OSKAR_IMAGE_TYPE_STOKES_Q ||
                        pol == OSKAR_IMAGE_TYPE_STOKES_U ||
                        pol == OSKAR_IMAGE_TYPE_STOKES_V)
                {
                    const double2* data = oskar_mem_double2_const(stokes, &err);
                    for (int i = 0, c = vis_chan_range[0]; c <= vis_chan_range[1]; ++c)
                    {
                        for (int t = vis_time_range[0]; t <=vis_time_range[1]; ++t)
                        {
                            int idx = (c*num_times+t)*num_baselines;
                            for (int b = 0; b < num_baselines;++b, ++i)
                            {
                                a_[i].x=data[idx+b].x; a_[i].y=data[idx+b].y;
                            }
                        }
                    }
                }
                else
                {
                    const double4c* data = oskar_mem_double4c_const(v_, &err);
                    for (int i = 0, c = vis_chan_range[0]; c <= vis_chan_range[1]; ++c)
                    {
                        for (int t = vis_time_range[0]; t <=vis_time_range[1]; ++t)
                        {
                            int idx = (c*num_times+t)*num_baselines;
                            for (int b = 0; b < num_baselines;++b, ++i)
                            {
                                if (pol == OSKAR_IMAGE_TYPE_POL_XX)
                                {
                                    a_[i].x=data[idx+b].a.x; a_[i].y=data[idx+b].a.y;
                                }
                                else if (pol == OSKAR_IMAGE_TYPE_POL_XY)
                                {
                                    a_[i].x=data[idx+b].b.x; a_[i].y=data[idx+b].b.y;
                                }
                                else if (pol == OSKAR_IMAGE_TYPE_POL_YX)
                                {
                                    a_[i].x=data[idx+b].c.x; a_[i].y=data[idx+b].c.y;
                                }
                                else if (pol == OSKAR_IMAGE_TYPE_POL_YY)
                                {
                                    a_[i].x=data[idx+b].d.x; a_[i].y=data[idx+b].d.y;
                                }
                            }
                        }
                    }
                }
            }
        }
    } /* [end] if (type == OSKAR_DOUBLE) */






    /* ================================================================= */
    /* ================================================================= */
    else /* single precision */
    {
        float2* a_ = oskar_mem_float2(amps, &err);

        /* ----------------------------------- TIME SNAPSHOTS, FREQ SNAPSHOTS */
        if (settings->time_snapshots && settings->channel_snapshots)
        {
            int idx = (vis_channel * num_times + vis_time) * num_baselines;
            if (num_pols == 4)
            {
                const float4c* data;
                if (pol == OSKAR_IMAGE_TYPE_POL_LINEAR)
                    data = oskar_mem_float4c_const(v_, &err);
                else /* pol == OSKAR_IMAGE_TYPE_STOKES */
                    data = oskar_mem_float4c_const(stokes, &err);

                for (int b = 0; b < num_baselines; ++b)
                {
                    if (p == 0)
                    {
                        a_[b].x = data[idx+b].a.x; a_[b].y= data[idx+b].a.y;
                    }
                    else if (p == 1)
                    {
                        a_[b].x=data[idx+b].b.x; a_[b].y=data[idx+b].b.y;
                    }
                    else if (p == 2)
                    {
                        a_[b].x=data[idx+b].c.x; a_[b].y=data[idx+b].c.y;
                    }
                    else if (p == 3)
                    {
                        a_[b].x=data[idx+b].d.x; a_[b].y=data[idx+b].d.y;
                    }
                }
            }
            else if (num_pols == 1)
            {
                if (pol == OSKAR_IMAGE_TYPE_STOKES_I ||
                        pol == OSKAR_IMAGE_TYPE_STOKES_Q ||
                        pol == OSKAR_IMAGE_TYPE_STOKES_U ||
                        pol == OSKAR_IMAGE_TYPE_STOKES_V)
                {
                    const float2* data = oskar_mem_float2_const(stokes, &err);
                    for (int b = 0; b < num_baselines; ++b)
                    {
                        a_[b].x = data[idx+b].x; a_[b].y = data[idx+b].y;
                    }
                }
                else
                {
                    const float4c* data = oskar_mem_float4c_const(v_, &err);
                    for (int b = 0; b < num_baselines; ++b)
                    {
                        if (pol == OSKAR_IMAGE_TYPE_POL_XX)
                        {
                            a_[b].x=data[idx+b].a.x; a_[b].y=data[idx+b].a.y;
                        }
                        else if (pol == OSKAR_IMAGE_TYPE_POL_XY)
                        {
                            a_[b].x=data[idx+b].b.x; a_[b].y=data[idx+b].b.y;
                        }
                        else if (pol == OSKAR_IMAGE_TYPE_POL_YX)
                        {
                            a_[b].x=data[idx+b].c.x; a_[b].y=data[idx+b].c.y;
                        }
                        else if (pol == OSKAR_IMAGE_TYPE_POL_YY)
                        {
                            a_[b].x=data[idx+b].d.x; a_[b].y=data[idx+b].d.y;
                        }
                    }
                }
            } /* num_pols == 1 */
        }



        /* ---------------------------------- TIME SNAPSHOTS, FREQ SYNTHESIS */
        else if (settings->time_snapshots && !settings->channel_snapshots)
        {
            if (num_pols == 4)
            {
                const float4c* data;
                if (pol == OSKAR_IMAGE_TYPE_POL_LINEAR)
                    data = oskar_mem_float4c_const(v_, &err);
                else /* pol == OSKAR_IMAGE_TYPE_STOKES */
                    data = oskar_mem_float4c_const(stokes, &err);
                for (int i = 0, c = vis_chan_range[0]; c <= vis_chan_range[1]; ++c)
                {
                    int idx = (c*num_times+vis_time)*num_baselines;
                    for (int b = 0; b < num_baselines;++b, ++i)
                    {
                        if (p == 0)
                        {
                            a_[i].x=data[idx+b].a.x; a_[i].y=data[idx+b].a.y;
                        }
                        else if (p == 1)
                        {
                            a_[i].x=data[idx+b].b.x; a_[i].y=data[idx+b].b.y;
                        }
                        else if (p == 2)
                        {
                            a_[i].x=data[idx+b].c.x; a_[i].y=data[idx+b].c.y;
                        }
                        else if (p == 3)
                        {
                            a_[i].x=data[idx+b].d.x; a_[i].y=data[idx+b].d.y;
                        }
                    }
                }
            }
            else /* (num_pols == 1) */
            {
                if (pol == OSKAR_IMAGE_TYPE_STOKES_I ||
                        pol == OSKAR_IMAGE_TYPE_STOKES_Q ||
                        pol == OSKAR_IMAGE_TYPE_STOKES_U ||
                        pol == OSKAR_IMAGE_TYPE_STOKES_V)
                {
                    const float2* data = oskar_mem_float2_const(stokes, &err);
                    for (int i = 0, c = vis_chan_range[0]; c <= vis_chan_range[1]; ++c)
                    {
                        int idx = (c*num_times+vis_time)*num_baselines;
                        for (int b = 0; b < num_baselines;++b, ++i)
                        {
                            a_[i].x=data[idx+b].x; a_[i].y=data[idx+b].y;
                        }
                    }
                }
                else
                {
                    const float4c* data = oskar_mem_float4c_const(v_, &err);
                    for (int i = 0, c = vis_chan_range[0]; c <= vis_chan_range[1]; ++c)
                    {
                        int idx = (c*num_times+vis_time)*num_baselines;
                        for (int b = 0; b < num_baselines;++b, ++i)
                        {
                            if (pol == OSKAR_IMAGE_TYPE_POL_XX)
                            {
                                a_[i].x=data[idx+b].a.x; a_[i].y=data[idx+b].a.y;
                            }
                            else if (pol == OSKAR_IMAGE_TYPE_POL_XY)
                            {
                                a_[i].x=data[idx+b].b.x; a_[i].y=data[idx+b].b.y;
                            }
                            else if (pol == OSKAR_IMAGE_TYPE_POL_YX)
                            {
                                a_[i].x=data[idx+b].c.x; a_[i].y=data[idx+b].c.y;
                            }
                            else if (pol == OSKAR_IMAGE_TYPE_POL_YY)
                            {
                                a_[i].x=data[idx+b].d.x; a_[i].y=data[idx+b].d.y;
                            }
                        }
                    }
                }
            } /* num_pols == 1 */
        }

        /* ---------------------------------- TIME SYNTHESIS, FREQ SNAPSHOTS */
        else if (!settings->time_snapshots && settings->channel_snapshots)
        {
            if (num_pols == 4)
            {
                const float4c* data;
                if (pol == OSKAR_IMAGE_TYPE_POL_LINEAR)
                    data = oskar_mem_float4c_const(v_, &err);
                else /* pol == OSKAR_IMAGE_TYPE_STOKES */
                    data = oskar_mem_float4c_const(stokes, &err);
                for (int i = 0, t = vis_time_range[0]; t <= vis_time_range[1]; ++t)
                {
                    int idx = (vis_channel*num_times+t)*num_baselines;
                    for (int b = 0; b < num_baselines;++b, ++i)
                    {
                        if (p == 0)
                        {
                            a_[i].x=data[idx+b].a.x; a_[i].y=data[idx+b].a.y;
                        }
                        else if (p == 1)
                        {
                            a_[i].x=data[idx+b].b.x; a_[i].y=data[idx+b].b.y;
                        }
                        else if (p == 2)
                        {
                            a_[i].x=data[idx+b].c.x; a_[i].y=data[idx+b].c.y;
                        }
                        else if (p == 3)
                        {
                            a_[i].x=data[idx+b].d.x; a_[i].y=data[idx+b].d.y;
                        }
                    }
                }
            }
            else /* (num_pols == 1) */
            {
                if (pol == OSKAR_IMAGE_TYPE_STOKES_I ||
                        pol == OSKAR_IMAGE_TYPE_STOKES_Q ||
                        pol == OSKAR_IMAGE_TYPE_STOKES_U ||
                        pol == OSKAR_IMAGE_TYPE_STOKES_V)
                {
                    const float2* data = oskar_mem_float2_const(stokes, &err);
                    for (int i = 0, t = vis_time_range[0]; t <= vis_time_range[1]; ++t)
                    {
                        int idx = (vis_channel*num_times+t)*num_baselines;
                        for (int b = 0; b < num_baselines;++b, ++i)
                        {
                            a_[i].x=data[idx+b].x; a_[i].y=data[idx+b].y;
                        }
                    }
                }
                else
                {
                    const float4c* data = oskar_mem_float4c_const(v_, &err);
                    for (int i = 0, t = vis_time_range[0]; t <= vis_time_range[1]; ++t)
                    {
                        int idx = (vis_channel*num_times+t)*num_baselines;
                        for (int b = 0; b < num_baselines;++b, ++i)
                        {
                            if (pol == OSKAR_IMAGE_TYPE_POL_XX)
                            {
                                a_[i].x=data[idx+b].a.x; a_[i].y=data[idx+b].a.y;
                            }
                            else if (pol == OSKAR_IMAGE_TYPE_POL_XY)
                            {
                                a_[i].x=data[idx+b].b.x; a_[i].y=data[idx+b].b.y;
                            }
                            else if (pol == OSKAR_IMAGE_TYPE_POL_YX)
                            {
                                a_[i].x=data[idx+b].c.x; a_[i].y=data[idx+b].c.y;
                            }
                            else if (pol == OSKAR_IMAGE_TYPE_POL_YY)
                            {
                                a_[i].x=data[idx+b].d.x; a_[i].y=data[idx+b].d.y;
                            }
                        }
                    }
                }
            } /* num_pols == 1 */
        }

        /* ----------------------------------- TIME SYNTHESIS, FREQ SYNTHESIS */
        else
        {
            if (num_pols == 4)
            {
                const float4c* data;
                if (pol == OSKAR_IMAGE_TYPE_POL_LINEAR)
                    data = oskar_mem_float4c_const(v_, &err);
                else /* pol == OSKAR_IMAGE_TYPE_STOKES */
                    data = oskar_mem_float4c_const(stokes, &err);
                for (int i = 0, c = vis_chan_range[0]; c <= vis_chan_range[1]; ++c)
                {
                    for (int t = vis_time_range[0]; t <=vis_time_range[1]; ++t)
                    {
                        int idx = (c*num_times+t)*num_baselines;
                        for (int b = 0; b < num_baselines;++b, ++i)
                        {
                            if (p == 0)
                            {
                                a_[i].x=data[idx+b].a.x; a_[i].y=data[idx+b].a.y;
                            }
                            else if (p == 1)
                            {
                                a_[i].x=data[idx+b].b.x; a_[i].y=data[idx+b].b.y;
                            }
                            else if (p == 2)
                            {
                                a_[i].x=data[idx+b].c.x; a_[i].y=data[idx+b].c.y;
                            }
                            else if (p == 3)
                            {
                                a_[i].x=data[idx+b].d.x; a_[i].y=data[idx+b].d.y;
                            }
                        }
                    }
                }
            }
            else
            {
                if (pol == OSKAR_IMAGE_TYPE_STOKES_I ||
                        pol == OSKAR_IMAGE_TYPE_STOKES_Q ||
                        pol == OSKAR_IMAGE_TYPE_STOKES_U ||
                        pol == OSKAR_IMAGE_TYPE_STOKES_V)
                {
                    const float2* data = oskar_mem_float2_const(stokes, &err);
                    for (int i = 0, c = vis_chan_range[0]; c <= vis_chan_range[1]; ++c)
                    {
                        for (int t = vis_time_range[0]; t <=vis_time_range[1]; ++t)
                        {
                            int idx = (c*num_times+t)*num_baselines;
                            for (int b = 0; b < num_baselines;++b, ++i)
                            {
                                a_[i].x=data[idx+b].x; a_[i].y=data[idx+b].y;
                            }
                        }
                    }
                }
                else
                {
                    const float4c* data = oskar_mem_float4c_const(v_, &err);
                    for (int i = 0, c = vis_chan_range[0]; c <= vis_chan_range[1]; ++c)
                    {
                        for (int t = vis_time_range[0]; t <=vis_time_range[1]; ++t)
                        {
                            int idx = (c*num_times+t)*num_baselines;
                            for (int b = 0; b < num_baselines;++b, ++i)
                            {
                                if (pol == OSKAR_IMAGE_TYPE_POL_XX)
                                {
                                    a_[i].x=data[idx+b].a.x; a_[i].y=data[idx+b].a.y;
                                }
                                else if (pol == OSKAR_IMAGE_TYPE_POL_XY)
                                {
                                    a_[i].x=data[idx+b].b.x; a_[i].y=data[idx+b].b.y;
                                }
                                else if (pol == OSKAR_IMAGE_TYPE_POL_YX)
                                {
                                    a_[i].x=data[idx+b].c.x; a_[i].y=data[idx+b].c.y;
                                }
                                else if (pol == OSKAR_IMAGE_TYPE_POL_YY)
                                {
                                    a_[i].x=data[idx+b].d.x; a_[i].y=data[idx+b].d.y;
                                }
                            }
                        }
                    }
                }
            }
        }
    } // [end] (type == OSKAR_SINGLE)

    return err;
}

#ifdef __cplusplus
}
#endif
