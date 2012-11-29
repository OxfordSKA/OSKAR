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

#include "math/oskar_random_broken_power_law.h"
#include "math/oskar_random_power_law.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

double oskar_random_broken_power_law(double min, double max, double breakpoint,
        double index1, double index2)
{
    double b0, pow1, pow2, powinv1, powinv2, b1, b2, r, b;
    b0 = pow(breakpoint, (index1 - index2));
    pow1 = index1 + 1.0;
    pow2 = index2 + 1.0;
    powinv1 = 1.0 / pow1;
    powinv2 = 1.0 / pow2;

    if ((min <= 0) || (max < min))
        return 0.0;

    if (min >= breakpoint) {
        return oskar_random_power_law(min, max, index1);
    }
    else if (max <= breakpoint) {
        return oskar_random_power_law(min, max, index2);
    }
    else {
        if (fabs(index1 + 1.0) < DBL_EPSILON) {
//            b1 = log(threshold) - log(min);
            b1 = log(breakpoint / min);
        }
        else {
            //b1 = powinv1 * (pow(breakpoint, pow1) - pow(min, pow2));
            b1 = (pow(breakpoint, index1+1) - pow(min, index1+1)) / (index1+1);
        }

        if (fabs(index2 + 1.0) < DBL_EPSILON) {
//            b2 = b0 * (log(max) - log(breakpoint));
            b2 = log(max / breakpoint) * pow(breakpoint, index1 - index2);
        }
        else {
//            b2 = b0 * powinv2 * (pow(max, pow2) - pow(breakpoint, pow2));
            b2 = (pow(max, index2 + 1) - pow(breakpoint, index2 + 1)) *
                    pow(breakpoint, index1 - index2) / (index2 + 1);
        }

        r = rand() / ((double)RAND_MAX + 1.0);

        if (r > b1 / (b1 + b2))
            return oskar_random_power_law(breakpoint, max, index2);
        else
            return oskar_random_power_law(min, breakpoint, index1);

//        r = (double)rand() / ((double)RAND_MAX + 1.0);
//        b = -b1 + r * (b2 + b1);
//
//        double value = 0.0;
//        if (b > 0.0)
//        {
//            if (index2 == -1.0)
//                value = breakpoint * exp(b / b0);
//            else
//                value =  pow((b * (pow2 / b0) + pow(breakpoint, pow2)), powinv2);
//        }
//        else
//        {
//            if (index1 == -1.0)
//                value = breakpoint * exp(-abs(b));
//            else
//                value = pow((pow(breakpoint, pow1) - abs(b) * pow1) , powinv1);
//        }
//        return value;
    }
}

#ifdef __cplusplus
}
#endif
