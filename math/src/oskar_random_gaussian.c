/*
 * Copyright (c) 2011-2015, The University of Oxford
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

#include <oskar_random_gaussian.h>
#include <private_random_helpers.h>
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

double oskar_random_gaussian(double* another)
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

#ifdef __cplusplus
}
#endif
