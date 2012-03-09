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


#include "math/oskar_rotate.h"
#include <stdlib.h>
#include <math.h>


#ifdef __cplusplus
extern "C" {
#endif

int oskar_rotate_x(int n, oskar_Mem* x, oskar_Mem* y, oskar_Mem* z,
        double angle)
{
    int i;

    if (x == NULL || y == NULL || z == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    if (x->location != OSKAR_LOCATION_CPU ||
            y->location != OSKAR_LOCATION_CPU ||
            z->location != OSKAR_LOCATION_CPU)
    {
        return OSKAR_ERR_BAD_LOCATION;
    }

    if (x->num_elements > n || y->num_elements > n || z->num_elements > n)
    {
        return OSKAR_ERR_DIMENSION_MISMATCH;
    }

    if (x->type == OSKAR_DOUBLE && y->type == OSKAR_DOUBLE &&
            z->type == OSKAR_DOUBLE)
    {
        double cosAng, sinAng;
        double y_, z_;
        cosAng = cos(angle);
        sinAng = sin(angle);
        for (i = 0; i < n; ++i)
        {
            y_ = ((double*)y->data)[i];
            z_ = ((double*)z->data)[i];
            ((double*)y->data)[i] = y_ * cosAng - z_ * sinAng;
            ((double*)z->data)[i] = y_ * sinAng + z_ * cosAng;
        }
    }
    else if (x->type == OSKAR_SINGLE && y->type == OSKAR_SINGLE &&
            z->type == OSKAR_SINGLE)
    {
        float cosAng, sinAng;
        float y_, z_;
        cosAng = cosf(angle);
        sinAng = sinf(angle);
        for (i = 0; i < n; ++i)
        {
            y_ = ((float*)y->data)[i];
            z_ = ((float*)z->data)[i];
            ((float*)y->data)[i] = y_ * cosAng - z_ * sinAng;
            ((float*)z->data)[i] = y_ * sinAng + z_ * cosAng;
        }
    }
    else
        return OSKAR_ERR_BAD_DATA_TYPE;


    return OSKAR_SUCCESS;
}



int oskar_rotate_y(int n, oskar_Mem* x, oskar_Mem* y, oskar_Mem* z,
        double angle)
{
    int i;

    if (x == NULL || y == NULL || z == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    if (x->location != OSKAR_LOCATION_CPU ||
            y->location != OSKAR_LOCATION_CPU ||
            z->location != OSKAR_LOCATION_CPU)
    {
        return OSKAR_ERR_BAD_LOCATION;
    }

    if (x->num_elements > n || y->num_elements > n || z->num_elements > n)
    {
        return OSKAR_ERR_DIMENSION_MISMATCH;
    }

    if (x->type == OSKAR_DOUBLE && y->type == OSKAR_DOUBLE &&
            z->type == OSKAR_DOUBLE)
    {
        double cosAng, sinAng;
        double x_, z_;
        cosAng = cos(angle);
        sinAng = sin(angle);
        for (i = 0; i < n; ++i)
        {
            x_ = ((double*)x->data)[i];
            z_ = ((double*)z->data)[i];
            ((double*)x->data)[i] =  x_ * cosAng + z_ * sinAng;
            ((double*)z->data)[i] = -x_ * sinAng + z_ * cosAng;
        }
    }
    else if (x->type == OSKAR_SINGLE && y->type == OSKAR_SINGLE &&
            z->type == OSKAR_SINGLE)
    {
        float cosAng, sinAng;
        float x_, z_;
        cosAng = cosf(angle);
        sinAng = sinf(angle);
        for (i = 0; i < n; ++i)
        {
            x_ = ((float*)x->data)[i];
            z_ = ((float*)z->data)[i];
            ((float*)x->data)[i] =  x_ * cosAng + z_ * sinAng;
            ((float*)z->data)[i] = -x_ * sinAng + z_ * cosAng;
        }
    }
    else
        return OSKAR_ERR_BAD_DATA_TYPE;

    return OSKAR_SUCCESS;
}



int oskar_rotate_z(int n, oskar_Mem* x, oskar_Mem* y, oskar_Mem* z,
        double angle)
{
    int i;

    if (x == NULL || y == NULL || z == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    if (x->location != OSKAR_LOCATION_CPU ||
            y->location != OSKAR_LOCATION_CPU ||
            z->location != OSKAR_LOCATION_CPU)
    {
        return OSKAR_ERR_BAD_LOCATION;
    }

    if (x->num_elements > n || y->num_elements > n || z->num_elements > n)
    {
        return OSKAR_ERR_DIMENSION_MISMATCH;
    }

    if (x->type == OSKAR_DOUBLE && y->type == OSKAR_DOUBLE &&
            z->type == OSKAR_DOUBLE)
    {
        double cosAng, sinAng;
        double x_, y_;
        cosAng = cos(angle);
        sinAng = sin(angle);
        for (i = 0; i < n; ++i)
        {
            x_ = ((double*)x->data)[i];
            y_ = ((double*)y->data)[i];
            ((double*)x->data)[i] = x_ * cosAng - y_ * sinAng;
            ((double*)y->data)[i] = x_ * sinAng + y_ * cosAng;
        }
    }
    else if (x->type == OSKAR_SINGLE && y->type == OSKAR_SINGLE &&
            z->type == OSKAR_SINGLE)
    {
        float cosAng, sinAng;
        float x_, y_;
        cosAng = cosf(angle);
        sinAng = sinf(angle);
        for (i = 0; i < n; ++i)
        {
            x_ = ((float*)x->data)[i];
            y_ = ((float*)y->data)[i];
            ((float*)x->data)[i] = x_ * cosAng - y_ * sinAng;
            ((float*)y->data)[i] = x_ * sinAng + y_ * cosAng;
        }
    }
    else
        return OSKAR_ERR_BAD_DATA_TYPE;

    return OSKAR_SUCCESS;
}


int oskar_rotate_sph(int n, oskar_Mem* x, oskar_Mem* y, oskar_Mem* z,
        double lon, double lat)
{
    int i;

    if (x == NULL || y == NULL || z == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    if (x->location != OSKAR_LOCATION_CPU ||
            y->location != OSKAR_LOCATION_CPU ||
            z->location != OSKAR_LOCATION_CPU)
    {
        return OSKAR_ERR_BAD_LOCATION;
    }

    if (x->num_elements > n || y->num_elements > n || z->num_elements > n)
    {
        return OSKAR_ERR_DIMENSION_MISMATCH;
    }

    if (x->type == OSKAR_DOUBLE && y->type == OSKAR_DOUBLE &&
            z->type == OSKAR_DOUBLE)
    {
        double cosLon, sinLon, cosLat, sinLat;
        double x_, y_, z_;
        cosLon = cos(lon);
        sinLon = sin(lon);
        cosLat = cos(lat);
        sinLat = sin(lat);
        for (i = 0; i < n; ++i)
        {
            x_ = ((double*)x->data)[i];
            y_ = ((double*)y->data)[i];
            z_ = ((double*)z->data)[i];
            ((double*)x->data)[i] = x_*cosLon*cosLat - y_*sinLon - z_*cosLon*sinLat;
            ((double*)y->data)[i] = x_*cosLat*sinLon + y_*cosLon - z_*sinLon*sinLat;
            ((double*)z->data)[i] = x_*sinLat + z_*cosLat;
        }
    }
    else if (x->type == OSKAR_SINGLE && y->type == OSKAR_SINGLE &&
            z->type == OSKAR_SINGLE)
    {
        float cosLon, sinLon, cosLat, sinLat;
        float x_, y_, z_;
        cosLon = cosf(lon);
        sinLon = sinf(lon);
        cosLat = cosf(lat);
        sinLat = sinf(lat);
        for (i = 0; i < n; ++i)
        {
            x_ = ((float*)x->data)[i];
            y_ = ((float*)y->data)[i];
            z_ = ((float*)z->data)[i];
            ((float*)x->data)[i] = x_*cosLon*cosLat - y_*sinLon - z_*cosLon*sinLat;
            ((float*)y->data)[i] = x_*cosLat*sinLon + y_*cosLon - z_*sinLon*sinLat;
            ((float*)z->data)[i] = x_*sinLat + z_*cosLat;
        }
    }
    else
        return OSKAR_ERR_BAD_DATA_TYPE;

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
