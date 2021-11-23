/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/oskar_random_gaussian.h"
#include "math/private_random_helpers.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_random_gaussian2(unsigned int seed, unsigned int counter0,
        unsigned int counter1, double rnd[2])
{
    OSKAR_R123_GENERATE_2(seed, counter0, counter1);
    oskar_box_muller_d(u.i[0], u.i[1], &rnd[0], &rnd[1]);
}

void oskar_random_gaussian4(unsigned int seed, unsigned int counter0,
        unsigned int counter1, unsigned int counter2, unsigned int counter3,
        double rnd[4])
{
    OSKAR_R123_GENERATE_4(seed, counter0, counter1, counter2, counter3);
    oskar_box_muller_d(u.i[0], u.i[1], &rnd[0], &rnd[1]);
    oskar_box_muller_d(u.i[2], u.i[3], &rnd[2], &rnd[3]);
}

#if 0
double oskar_random_gaussian(double* another)
{
    double x = 0.0, y = 0.0, r2 = 0.0, fac = 0.0;
    do
    {
        /* Choose x and y in a uniform square (-1, -1) to (+1, +1). */
        x = 2.0 * rand() / (RAND_MAX + 1.0) - 1.0;
        y = 2.0 * rand() / (RAND_MAX + 1.0) - 1.0;

        /* Check if this is in the unit circle. */
        r2 = x*x + y*y;
    } while (r2 >= 1.0 || r2 == 0.0);

    /* Box-Muller transform. */
    fac = sqrt(-2.0 * log(r2) / r2);
    x *= fac;
    if (another) *another = y * fac;

    /* Return the first random number. */
    return x;
}
#endif

#ifdef __cplusplus
}
#endif
