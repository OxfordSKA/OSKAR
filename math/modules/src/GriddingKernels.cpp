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

#include "math/modules/GriddingKernels.h"
#include "math/core/FloatingPointCompare.h"

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

namespace oskar {

float GriddingKernels::exp1D(const float r2, const float sigma)
{
    return 1.0f;
}


void GriddingKernels::exp2D(const unsigned support,
        const unsigned oversample, const float sigma, float * cFunc)
{
    const int size = (2 * support + 1) * oversample;
    const int centre = (size - 1) / 2;
    const float inc = 1.0f / (float)oversample;
    const float p1 = 1.0f / (2.0f * sigma * sigma);

    for (int j = 0; j < size; ++j)
    {
        for (int i = 0; i < size; ++i)
        {
            const float y = float(j - centre) * inc;
            const float x = float(i - centre) * inc;
            const float r2 = x * x + y * y;

            cFunc[j * size + i] = exp(-r2 / p1);
        }
    }
}



float GriddingKernels::expSinc1D(const float r)
{
    return 1.0f;
}



void GriddingKernels::expSinc2D(const unsigned support,
        const unsigned oversample, float * cFunc)
{
    const int size = (2 * support + 1) * oversample;
    const float inc = 1.0f / (float)oversample;
    const int centre = (size - 1) / 2;

    for (int j = 0; j < size; ++j)
    {
        for (int i = 0; i < size; ++i)
        {
            const float y = (j - centre) * inc;
            const float x = (i - centre) * inc;
            cFunc[j * size + i] = _expSinc(x, y);
        }
    }
}


float GriddingKernels::_expSinc(const float x, const float y)
{
    const float p1 = M_PI / 1.55f;
    const float p2 = 1.0f / 2.52f;
    const float p3 = 2.0f;

    const float x2 = pow((fabs(x) * p2), p3);
    const float y2 = pow((fabs(y) * p2), p3);
    const float r2 = x2 + y2;
    const float ampExp = exp(-r2);

    float ampSinc = 0.0f;

    if ( isEqual(x, 0.0f) && isEqual(y, 0.0f) )
    {
        ampSinc = 1.0f;
    }
    else if (isEqual(x, 0.0f))
    {
        ampSinc = sin(y * p1) / (y * p1);
    }
    else if (isEqual(y, 0.0f))
    {
        ampSinc = sin(x * p1) / (x * p1);
    }
    else
    {
        ampSinc = sin(x * p1) * sin(y * p1);
        ampSinc /= (x * p1 * y * p1);
    }

    return (ampExp * ampSinc);
}



} // namespace oskar
