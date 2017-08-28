/*
 * Copyright (c) 2017, The University of Oxford
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

#include "imager/private_imager_composite_nearest_even.h"

#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

static int cmpfunc(const void* a, const void* b)
{
    return (*(const int*)a - *(const int*)b);
}


int oskar_imager_composite_nearest_even(int value, int* smaller, int *larger)
{
    double x;
    int i = 0, i2, i3, i5, n2, n3, n5, nt, *values, up = 0, down = 0;
    x = (double) value;
    n2 = 1 + (int) (log(x) / log(2.0) + 1.0);
    n3 = 1 + (int) (log(x) / log(3.0) + 1.0);
    n5 = 1 + (int) (log(x) / log(5.0) + 1.0);
    nt = n2 * n3 * n5;
    values = (int*) malloc(nt * sizeof(int));
    for (i2 = 0; i2 < n2; ++i2)
    {
        for (i3 = 0; i3 < n3; ++i3)
        {
            for (i5 = 0; i5 < n5; ++i5, ++i)
            {
                values[i] = (int) round(
                        pow(2.0, (double) i2) *
                        pow(3.0, (double) i3) *
                        pow(5.0, (double) i5));
            }
        }
    }
    qsort(values, nt, sizeof(int), cmpfunc);

    /* Get next larger even. */
    for (i = 0; i < nt; ++i)
    {
        up = values[i];
        if ((up > value) && (up % 2 == 0)) break;
    }

    /* Get next smaller even. */
    for (i = nt - 1; i >= 0; --i)
    {
        down = values[i];
        if ((down < value) && (down % 2 == 0)) break;
    }

    free(values);
    if (smaller) *smaller = down;
    if (larger) *larger = up;
    return (abs(up - value) < abs(down - value) ? up : down);
}


#ifdef __cplusplus
}
#endif
