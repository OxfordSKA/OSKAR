/*
 * Copyright (c) 2015, The University of Oxford
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

#include <gtest/gtest.h>

#include <oskar_mem.h>
#include <oskar_timer.h>
#include <oskar_random_gaussian.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

static const bool verbose = false;
static const bool save = false;

static void report_time(int n, const char* type,
        const char* prec, const char* loc, double sec)
{
    if (verbose)
        printf("Generated %d %s random numbers (%s, %s): %.3f sec\n",
                n, type, prec, loc, sec);
}

static double random_gaussian_old(double* another)
{
    double x, y, r2, fac;
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
    double *t;
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
        double another;
        t[i] = random_gaussian_old(&another);
        t[i + 1] = another;
    }
    report_time(n, "Gaussian[2]", "double", "OLD", oskar_timer_elapsed(tmr));

    if (save)
    {
        FILE* fhan = fopen("random_gaussian.txt", "w");
        oskar_mem_save_ascii(fhan, 3, n, &status, a, data2, data4);
        fclose(fhan);
    }

    // Free memory.
    oskar_mem_free(a, &status);
    oskar_mem_free(data2, &status);
    oskar_mem_free(data4, &status);
    oskar_timer_free(tmr);
}
