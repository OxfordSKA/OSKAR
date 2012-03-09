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


#include "math/oskar_cart2sph.h"
#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_cart2sph(int n, oskar_Mem* lon, oskar_Mem* lat,
        oskar_Mem* x, oskar_Mem* y, oskar_Mem* z)
{
    int i;
    double x_, y_, z_;

    if (x == NULL || y == NULL || z == NULL || lon == NULL || lat == NULL)
    {
        return OSKAR_ERR_INVALID_ARGUMENT;
    }

    if (x->location != OSKAR_LOCATION_CPU ||
            y->location != OSKAR_LOCATION_CPU ||
            z->location != OSKAR_LOCATION_CPU ||
            lon->location != OSKAR_LOCATION_CPU ||
            lat->location != OSKAR_LOCATION_CPU)
    {
        return OSKAR_ERR_BAD_LOCATION;
    }

    if (x->num_elements > n || y->num_elements > n || z->num_elements > n ||
            lon->num_elements > n || lat->num_elements > n)
    {
        return OSKAR_ERR_OUT_OF_RANGE;
    }

    if (x->type == OSKAR_DOUBLE && y->type == OSKAR_DOUBLE &&
            z->type == OSKAR_DOUBLE && lon->type == OSKAR_DOUBLE &&
            lat->type == OSKAR_DOUBLE)
    {
        for (i = 0; i < n; ++i)
        {
            x_ = ((double*)x->data)[i];
            y_ = ((double*)y->data)[i];
            z_ = ((double*)z->data)[i];

            ((double*)lon->data)[i] = atan2(y_,x_);
            ((double*)lat->data)[i] = atan2(z_, sqrt(x_*x_ + y_*y_));
        }
    }
    else if (x->type == OSKAR_SINGLE && y->type == OSKAR_SINGLE &&
            z->type == OSKAR_SINGLE && lon->type == OSKAR_SINGLE &&
            lat->type == OSKAR_SINGLE)
    {
        for (i = 0; i < n; ++i)
        {
            x_ = ((float*)x->data)[i];
            y_ = ((float*)y->data)[i];
            z_ = ((float*)z->data)[i];

            ((float*)lon->data)[i] = atan2(y_,x_);
            ((float*)lat->data)[i] = atan2(z_, sqrt(x_*x_ + y_*y_));
        }
    }
    else
        return OSKAR_ERR_BAD_DATA_TYPE;

    return OSKAR_SUCCESS;
}


#ifdef __cplusplus
}
#endif
