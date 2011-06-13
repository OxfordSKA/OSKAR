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

#include "imaging/ConvFunc.h"
#include "imaging/floating_point_compare.h"

#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

namespace oskar {


ConvFunc::ConvFunc()
: _size(0), _support(0), _oversample(0)
{
}


ConvFunc::~ConvFunc()
{
}



void ConvFunc::pillbox(const unsigned support, const unsigned oversample,
        const float width)
{
    const unsigned size = (support * 2 + 1) * oversample;
    const int centre = size / 2;
    const float inc = 1.0f / static_cast<float>(oversample);

    if (_size != size)
    {
        _convFunc.resize(size);
        _size =  size;
    }

    float * amp = &_convFunc[0];
    _oversample = oversample;
    _support = support;

    for (int i = 0; i < static_cast<int>(size); ++i)
    {
        const float x = static_cast<float>(i - centre) * inc;
        const float abs_x = fabs(x);

        if (abs_x > width)
            amp[i] = 0.0f;

        else if (isEqual<float>(abs_x, width))
            amp[i] = 0.5f;

        else
            amp[i] = 1.0f;
    }

}



void ConvFunc::exp(const unsigned support, const unsigned oversample)
{
    const unsigned size = (support * 2 + 1) * oversample;
    const float inc = 1.0f / static_cast<float>(oversample);
    const int centre = size / 2;

    if (_size != size)
    {
        _convFunc.resize(size);
        _size =  size;
    }
    _oversample = oversample;
    _support = support;

    float * amp = &_convFunc[0];

    // AIPS 'CONVFN.FOR' values.
    const float p1 = 1.0f / 1.55f;
    const float p2 = 2.52f;

    for (int i = 0; i < static_cast<int>(size); ++i)
    {
        const float x = static_cast<float>(i - centre) * inc;
        const float x2 = pow((fabs(x) * p1), p2);
        amp[i] = std::exp(-x2);
    }
}




void ConvFunc::sinc(const unsigned support, const unsigned oversample)
{
    const unsigned size = (support * 2 + 1) * oversample;
    const float inc = 1.0f / static_cast<float>(oversample);
    const int centre = size / 2;
    const float x_max = 3.0f; // AIPS = 3.0f

    if (_size != size)
    {
        _convFunc.resize(size, 0.0f);
        _size =  size;
    }

    float * amp = &_convFunc[0];
    _oversample = oversample;
    _support = support;

    const float p1 = 1.0f / 1.55f;

    for (int i = 0; i < static_cast<int>(size); ++i)
    {
        const float x = static_cast<float>(i - centre) * inc;
        const float abs_x = fabs(x);

        if (isEqual<float>(abs_x, 0.0f))
            amp[i] = 1.0f;

        else if (abs_x < x_max)
        {
            const float arg = p1 * abs_x;
            amp[i] = sin(arg) / arg;
        }
    }
}



void ConvFunc::expSinc(const unsigned support, const unsigned oversample)
{
    const unsigned size = (support * 2 + 1) * oversample;
    const float inc = 1.0f / static_cast<float>(oversample);
    const int centre = size / 2;
    const float x_max = 3.0f; // AIPS = 3.0f

    if (_size != size)
    {
        _convFunc.resize(size, 0.0f);
        _size =  size;
    }

    float * amp = &_convFunc[0];
    _oversample = oversample;
    _support = support;

    const float p1 = M_PI / 1.55f;
    const float p2 = 1.0f / 2.52f;
    const float p3 = 2.0f;

    for (int i = 0; i < static_cast<int>(size); ++i)
    {
        const float x = static_cast<float>(i - centre) * inc;
        const float abs_x = fabs(x);

        if (abs_x < inc)
        {
            amp[i] = 1.0f;
        }

        else if (abs_x < x_max)
        {
            const float arg = p1 * abs_x;
            const float ampSinc = sin(arg) / arg;
            const float ampExp = std::exp(-pow((fabs(x) * p2), p3));
            amp[i] = ampExp * ampSinc;
        }
    }
}


void ConvFunc::spherodial()
{
    // TODO!
}


void ConvFunc::makeConvFuncImage()
{
    std::vector<float> temp(_size * _size);
    float * t = &temp[0];
    float * c = &_convFunc[0];
    for (unsigned j = 0; j < _size; ++j)
    {
        for (unsigned i = 0; i < _size; ++i)
        {
            t[j * _size + i] = c[j] * c[i];
        }
    }
    _convFunc.resize(_size * _size);

    memcpy((void*)&_convFunc[0], (const void*)&temp[0],
            temp.size() * sizeof(float));
}

} // namespace oskar
