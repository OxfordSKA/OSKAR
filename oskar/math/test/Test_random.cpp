/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "math/oskar_random_gaussian.h"
#include "mem/oskar_mem.h"
#include "utility/oskar_timer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

static const bool verbose = false;
static const bool save = false;

static void report_time(int n, const char* type,
        const char* prec, const char* loc, double sec)
{
    if (verbose)
    {
        printf("Generated %d %s random numbers (%s, %s): %.3f sec\n",
                n, type, prec, loc, sec);
    }
}

static double random_gaussian_old(double* another)
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

TEST(random_gaussian, random_gaussian24)
{
    int seed = 1;
    int n = 256 * 10240;
    int half = n / 2;
    int quarter = half / 2;
    int status = 0;
    double *t = 0;
    oskar_Mem* a     = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, n, &status);
    oskar_Mem* data2 = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, n, &status);
    oskar_Mem* data4 = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, n, &status);
    oskar_Timer* tmr = oskar_timer_create(OSKAR_TIMER_NATIVE);

    // Run 2 at once, repeatedly.
    t = oskar_mem_double(data2, &status);
    oskar_timer_start(tmr);
#pragma omp parallel for
    for (int i = 0; i < half; ++i)
    {
        oskar_random_gaussian2(seed, i, 0, &t[2*i]);
    }
    report_time(n, "Gaussian[2]", "double", "OMP", oskar_timer_elapsed(tmr));

    // Run 4 at once, repeatedly.
    t = oskar_mem_double(data4, &status);
    oskar_timer_start(tmr);
#pragma omp parallel for
    for (int i = 0; i < quarter; ++i)
    {
        oskar_random_gaussian4(seed, i, 0, 0, 0, &t[4*i]);
    }
    report_time(n, "Gaussian[4]", "double", "OMP", oskar_timer_elapsed(tmr));

    // Old method (can only be single threaded).
    srand(seed);
    t = oskar_mem_double(a, &status);
    oskar_timer_start(tmr);
    for (int i = 0; i < n; i += 2)
    {
        double another = 0.0;
        t[i] = random_gaussian_old(&another);
        t[i + 1] = another;
    }
    report_time(n, "Gaussian[2]", "double", "OLD", oskar_timer_elapsed(tmr));

    if (save)
    {
        FILE* fhan = fopen("random_gaussian.txt", "w");
        oskar_mem_save_ascii(fhan, 3, 0, n, &status, a, data2, data4);
        fclose(fhan);
    }

    // Free memory.
    oskar_mem_free(a, &status);
    oskar_mem_free(data2, &status);
    oskar_mem_free(data4, &status);
    oskar_timer_free(tmr);
}
