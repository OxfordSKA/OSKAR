/*
 * Copyright (c) 2017-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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
    double x = 0.0;
    int i = 0, i2 = 0, i3 = 0, i5 = 0, n2 = 0, n3 = 0, n5 = 0, nt = 0;
    int *values = 0, up = 0, down = 0;
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
