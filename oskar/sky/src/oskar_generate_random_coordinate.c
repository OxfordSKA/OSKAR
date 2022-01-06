/*
 * Copyright (c) 2012-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/oskar_generate_random_coordinate.h"
#include "math/oskar_cmath.h"
#include <stdlib.h>

void oskar_generate_random_coordinate(double* longitude, double* latitude)
{
    double r1 = 0.0, r2 = 0.0;

    /* NOLINTNEXTLINE: We can use rand() here without concern. */
    r1 = (double)rand() / ((double)RAND_MAX + 1.0);
    /* NOLINTNEXTLINE: We can use rand() here without concern. */
    r2 = (double)rand() / ((double)RAND_MAX + 1.0);
    *latitude = M_PI / 2.0 - acos(2.0 * r1 - 1);
    *longitude  = 2.0 * M_PI * r2;
}
