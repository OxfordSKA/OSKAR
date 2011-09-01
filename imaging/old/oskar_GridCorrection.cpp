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

#include "imaging/oskar_GridCorrection.h"
#include "imaging/oskar_StandardGridKernels.h"
#include "imaging/floating_point_compare.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <limits>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

void GridCorrection::computeCorrection(oskar_StandardGridKernels & c,
        const unsigned grid_size)
{
    const unsigned c_size = c.size();
    const unsigned c_oversample = c.oversample();
    const float * convFunc = c.values();
    vector<float> c_x(c_size);
    const unsigned c_centre = c_size / 2;
    const float c_inc = 1.0f / c_oversample;

    const int g_centre = grid_size / 2;
    const float g_inc = 1.0f / static_cast<float>(grid_size);

//    printf("c_size       = %d\n", c_size);
//    printf("c_centre     = %d\n", c_centre);
//    printf("c_oversample = %d\n", c_oversample);
//    printf("c_inc        = %f\n", c_inc);

    for (unsigned i = 0; i < c_size; ++i)
    {
        c_x[i] = (float(i) - float(c_centre)) * c_inc;
//        printf("c_x = %f\n", c_x[i]);
    }

    _correction.resize(grid_size, 0.0f);
    _size = grid_size;
    float * correction = &_correction[0];

    // Sinc term sinc(pi * f * x) / (pi * f * x).
    vector<float> fsinc(grid_size);
    const float f = 1.0f / float(c_oversample);
    for (int i = 0; i < static_cast<int>(grid_size); ++i)
    {
        const float x = (static_cast<float>(i - g_centre)) * g_inc;
        const float abs_x = fabs(x);
//        printf("x = %f %f\n", x, abs_x);

        if (isEqual<float>(abs_x, 0.0f))
            fsinc[i] = 1.0f;
        else
        {
            const float arg = M_PI * f * abs_x;
            fsinc[i] = sin(arg) / arg;
        }
    }

    // DFT of convolution function.
    vector<float> csinc(grid_size);
    for (unsigned i = 0; i < grid_size; ++i)
    {
        const float x = (((float)i - g_centre)) / float(grid_size);
        for (unsigned j = 0; j < c_size; ++j)
        {
            const float arg = -2 * M_PI * x * c_x[j];
            csinc[i] += convFunc[j] * cos(arg);
        }
    }

    for (unsigned i = 0; i < grid_size; ++i)
        correction[i] = fsinc[i] * csinc[i];

    float max = findMax();
    for (unsigned i = 0; i < grid_size; ++i)
    {
        correction[i] /= max;
    }
}

void GridCorrection::make2D()
{
    vector<float> temp(_size * _size);
    float * t = &temp[0];
    float * c = &_correction[0];
    for (unsigned j = 0; j < _size; ++j)
    {
        for (unsigned i = 0; i < _size; ++i)
        {
            t[j * _size + i] = c[j] * c[i];
        }
    }
    _correction.resize(_size * _size);
    memcpy((void*)&_correction[0], (const void*)t, _size * _size * sizeof(float));
}


float GridCorrection::findMax()
{
    float convmax = -numeric_limits<float>::max();
//    for (unsigned i = 0; i < _size * _size; ++i)
//        convmax = max(convmax, abs(_correction[i]));
    for (unsigned i = 0; i < _size; ++i)
        convmax = max(convmax, abs(_correction[i]));
    return convmax;
}

