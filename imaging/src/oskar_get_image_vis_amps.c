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


#include "imaging/oskar_get_image_vis_amps.h"

#include "utility/oskar_mem_type_check.h"
#include "utility/oskar_vector_types.h"
#include <stdio.h>


#ifdef __cplusplus
extern "C" {
#endif

int oskar_get_image_vis_amps(oskar_Mem* amps, const oskar_Visibilities* vis,
        const oskar_Mem* stokes, const oskar_SettingsImage* settings,
        int channel, int time, int p)
{
    int type;
    int pol;
    int num_pols; /* number of image pols */
    int idx;
    int i, j, k, t;
    int chan_range[2];
    int time_range[2];

    /* Set local variables */
    type = oskar_mem_is_double(vis->amplitude.type) ? OSKAR_DOUBLE : OSKAR_SINGLE;
    pol = settings->polarisation;
    num_pols = (pol ==  OSKAR_IMAGE_TYPE_STOKES ||
            pol == OSKAR_IMAGE_TYPE_POL_LINEAR) ? 4 : 1;
    chan_range[0] = settings->channel_range[0];
    chan_range[1] = settings->channel_range[1];
    time_range[0] = settings->time_range[0];
    time_range[1] = settings->time_range[1];
    if (time_range[1] > vis->num_times-1) return OSKAR_ERR_OUT_OF_RANGE;
    if (time_range[0] < 0) time_range[0] = 0;
    if (time_range[1] < 0) time_range[1] = vis->num_times-1;
    if (chan_range[1] > vis->num_channels-1) return OSKAR_ERR_OUT_OF_RANGE;
    if (chan_range[0] < 0) chan_range[0] = 0;
    if (chan_range[1] < 0) chan_range[1] = vis->num_channels-1;


    /* ================================================================= */
    if (type == OSKAR_DOUBLE)
    {
        double2* a_ = (double2*)amps->data;

        /* ----------------------------------- TIME SNAPSHOTS, FREQ SNAPSHOTS */
        if (settings->time_snapshots && settings->channel_snapshots)
        {
            idx = (channel * vis->num_times + time) * vis->num_baselines;
            if (num_pols == 4)
            {
                double4c* data;
                if (pol == OSKAR_IMAGE_TYPE_POL_LINEAR)
                    data = (double4c*)vis->amplitude.data;
                else /* pol == OSKAR_IMAGE_TYPE_STOKES */
                    data = (double4c*)stokes->data;

                for (i = 0; i < vis->num_baselines; ++i)
                {
                    if (p == 0)
                    {
                        a_[i].x = data[idx+i].a.x; a_[i].y= data[idx+i].a.y;
                    }
                    else if (p == 1)
                    {
                        a_[i].x=data[idx+i].b.x; a_[i].y=data[idx+i].b.y;
                    }
                    else if (p == 2)
                    {
                        a_[i].x=data[idx+i].c.x; a_[i].y=data[idx+i].c.y;
                    }
                    else if (p == 3)
                    {
                        a_[i].x=data[idx+i].d.x; a_[i].y=data[idx+i].d.y;
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
                    double2* data = (double2*)stokes->data;
                    for (i = 0; i < vis->num_baselines; ++i)
                    {
                        a_[i].x = data[idx+i].x; a_[i].y = data[idx+i].y;
                    }
                }
                else
                {
                    double4c* data = (double4c*)vis->amplitude.data;
                    for (i = 0; i < vis->num_baselines; ++i)
                    {
                        if (pol == OSKAR_IMAGE_TYPE_POL_XX)
                        {
                            a_[i].x=data[idx+i].a.x; a_[i].y=data[idx+i].a.y;
                        }
                        else if (pol == OSKAR_IMAGE_TYPE_POL_XY)
                        {
                            a_[i].x=data[idx+i].b.x; a_[i].y=data[idx+i].b.y;
                        }
                        else if (pol == OSKAR_IMAGE_TYPE_POL_YX)
                        {
                            a_[i].x=data[idx+i].c.x; a_[i].y=data[idx+i].c.y;
                        }
                        else if (pol == OSKAR_IMAGE_TYPE_POL_YY)
                        {
                            a_[i].x=data[idx+i].d.x; a_[i].y=data[idx+i].d.y;
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
                double4c* data;
                if (pol == OSKAR_IMAGE_TYPE_POL_LINEAR)
                    data = (double4c*)vis->amplitude.data;
                else /* pol == OSKAR_IMAGE_TYPE_STOKES */
                    data = (double4c*)stokes->data;
                for (k = 0, j = chan_range[0]; j <= chan_range[1]; ++j)
                {
                    idx = (j*vis->num_times+time)*vis->num_baselines;
                    for (i = 0; i < vis->num_baselines;++i, ++k)
                    {
                        if (p == 0)
                        {
                            a_[k].x=data[idx+i].a.x; a_[k].y=data[idx+i].a.y;
                        }
                        else if (p == 1)
                        {
                            a_[k].x=data[idx+i].b.x; a_[k].y=data[idx+i].b.y;
                        }
                        else if (p == 2)
                        {
                            a_[k].x=data[idx+i].c.x; a_[k].y=data[idx+i].c.y;
                        }
                        else if (p == 3)
                        {
                            a_[k].x=data[idx+i].d.x; a_[k].y=data[idx+i].d.y;
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
                    double2* data = (double2*)stokes->data;
                    for (k = 0, j = chan_range[0]; j <= chan_range[1]; ++j)
                    {
                        idx = (j*vis->num_times+time)*vis->num_baselines;
                        for (i = 0; i < vis->num_baselines;++i, ++k)
                        {
                            a_[k].x=data[idx+i].x; a_[k].y=data[idx+i].y;
                        }
                    }
                }
                else
                {
                    double4c* data = (double4c*)vis->amplitude.data;
                    for (k = 0, j = chan_range[0]; j <= chan_range[1]; ++j)
                    {
                        idx = (j*vis->num_times+time)*vis->num_baselines;
                        for (i = 0; i < vis->num_baselines;++i, ++k)
                        {
                            if (pol == OSKAR_IMAGE_TYPE_POL_XX)
                            {
                                a_[k].x=data[idx+i].a.x; a_[k].y=data[idx+i].a.y;
                            }
                            else if (pol == OSKAR_IMAGE_TYPE_POL_XY)
                            {
                                a_[k].x=data[idx+i].b.x; a_[k].y=data[idx+i].b.y;
                            }
                            else if (pol == OSKAR_IMAGE_TYPE_POL_YX)
                            {
                                a_[k].x=data[idx+i].c.x; a_[k].y=data[idx+i].c.y;
                            }
                            else if (pol == OSKAR_IMAGE_TYPE_POL_YY)
                            {
                                a_[k].x=data[idx+i].d.x; a_[k].y=data[idx+i].d.y;
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
                double4c* data;
                if (pol == OSKAR_IMAGE_TYPE_POL_LINEAR)
                    data = (double4c*)vis->amplitude.data;
                else /* pol == OSKAR_IMAGE_TYPE_STOKES */
                    data = (double4c*)stokes->data;
                for (k = 0, j = time_range[0]; j <= time_range[1]; ++j)
                {
                    idx = (channel*vis->num_times+j)*vis->num_baselines;
                    for (i = 0; i < vis->num_baselines;++i, ++k)
                    {
                        if (p == 0)
                        {
                            a_[k].x=data[idx+i].a.x; a_[k].y=data[idx+i].a.y;
                        }
                        else if (p == 1)
                        {
                            a_[k].x=data[idx+i].b.x; a_[k].y=data[idx+i].b.y;
                        }
                        else if (p == 2)
                        {
                            a_[k].x=data[idx+i].c.x; a_[k].y=data[idx+i].c.y;
                        }
                        else if (p == 3)
                        {
                            a_[k].x=data[idx+i].d.x; a_[k].y=data[idx+i].d.y;
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
                    double2* data = (double2*)stokes->data;
                    for (k = 0, j = time_range[0]; j <= time_range[1]; ++j)
                    {
                        idx = (channel*vis->num_times+j)*vis->num_baselines;
                        for (i = 0; i < vis->num_baselines;++i, ++k)
                        {
                            a_[k].x=data[idx+i].x; a_[k].y=data[idx+i].y;
                        }
                    }
                }
                else
                {
                    double4c* data = (double4c*)vis->amplitude.data;
                    for (k = 0, j = time_range[0]; j <= time_range[1]; ++j)
                    {
                        idx = (channel*vis->num_times+j)*vis->num_baselines;
                        for (i = 0; i < vis->num_baselines;++i, ++k)
                        {
                            if (pol == OSKAR_IMAGE_TYPE_POL_XX)
                            {
                                a_[k].x=data[idx+i].a.x; a_[k].y=data[idx+i].a.y;
                            }
                            else if (pol == OSKAR_IMAGE_TYPE_POL_XY)
                            {
                                a_[k].x=data[idx+i].b.x; a_[k].y=data[idx+i].b.y;
                            }
                            else if (pol == OSKAR_IMAGE_TYPE_POL_YX)
                            {
                                a_[k].x=data[idx+i].c.x; a_[k].y=data[idx+i].c.y;
                            }
                            else if (pol == OSKAR_IMAGE_TYPE_POL_YY)
                            {
                                a_[k].x=data[idx+i].d.x; a_[k].y=data[idx+i].d.y;
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
                double4c* data;
                if (pol == OSKAR_IMAGE_TYPE_POL_LINEAR)
                    data = (double4c*)vis->amplitude.data;
                else /* pol == OSKAR_IMAGE_TYPE_STOKES */
                    data = (double4c*)stokes->data;
                for (k = 0, j = chan_range[0]; j <= chan_range[1]; ++j)
                {
                    for (t = time_range[0]; t <=time_range[1]; ++t)
                    {
                        idx = (j*vis->num_times+t)*vis->num_baselines;
                        for (i = 0; i < vis->num_baselines;++i, ++k)
                        {
                            if (p == 0)
                            {
                                a_[k].x=data[idx+i].a.x; a_[k].y=data[idx+i].a.y;
                            }
                            else if (p == 1)
                            {
                                a_[k].x=data[idx+i].b.x; a_[k].y=data[idx+i].b.y;
                            }
                            else if (p == 2)
                            {
                                a_[k].x=data[idx+i].c.x; a_[k].y=data[idx+i].c.y;
                            }
                            else if (p == 3)
                            {
                                a_[k].x=data[idx+i].d.x; a_[k].y=data[idx+i].d.y;
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
                    double2* data = (double2*)stokes->data;
                    for (k = 0, j = chan_range[0]; j <= chan_range[1]; ++j)
                    {
                        for (t = time_range[0]; t <=time_range[1]; ++t)
                        {
                            idx = (j*vis->num_times+t)*vis->num_baselines;
                            for (i = 0; i < vis->num_baselines;++i, ++k)
                            {
                                a_[k].x=data[idx+i].x; a_[k].y=data[idx+i].y;
                            }
                        }
                    }
                }
                else
                {
                    double4c* data = (double4c*)vis->amplitude.data;
                    for (k = 0, j = chan_range[0]; j <= chan_range[1]; ++j)
                    {
                        for (t = time_range[0]; t <=time_range[1]; ++t)
                        {
                            idx = (j*vis->num_times+t)*vis->num_baselines;
                            for (i = 0; i < vis->num_baselines;++i, ++k)
                            {
                                if (pol == OSKAR_IMAGE_TYPE_POL_XX)
                                {
                                    a_[k].x=data[idx+i].a.x; a_[k].y=data[idx+i].a.y;
                                }
                                else if (pol == OSKAR_IMAGE_TYPE_POL_XY)
                                {
                                    a_[k].x=data[idx+i].b.x; a_[k].y=data[idx+i].b.y;
                                }
                                else if (pol == OSKAR_IMAGE_TYPE_POL_YX)
                                {
                                    a_[k].x=data[idx+i].c.x; a_[k].y=data[idx+i].c.y;
                                }
                                else if (pol == OSKAR_IMAGE_TYPE_POL_YY)
                                {
                                    a_[k].x=data[idx+i].d.x; a_[k].y=data[idx+i].d.y;
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
    /* ================================================================= */
    /* ================================================================= */
    else /* (type == OSKAR_SINGLE) */
    {
        return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    }


    return OSKAR_SUCCESS;
}


#ifdef __cplusplus
}
#endif
