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

#include "imaging/WProjConvFunc.h"
#include "imaging/FFTUtility.h"

#include "fftw3.h"

#include <cmath>
#include <cassert> // disable asserts with #define NDEBUG before this include.
#include <cstdio>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

namespace oskar {

WProjConvFunc::WProjConvFunc()
: _size(0)
{
}

WProjConvFunc::~WProjConvFunc()
{
}


void WProjConvFunc::generateLM(const unsigned innerSize, const unsigned padding,
        const float pixelSizeLM_rads, const float w, const float taperFactor)
{
    // Resize the function if needed.
    _size = innerSize * padding;
    if (_convFunc.size() != _size * _size)
        _convFunc.resize(_size * _size, Complex(0.0, 0.0));

    // Generate the LM plane phase screen.
    _wFuncLMPadded(innerSize, _size, pixelSizeLM_rads, w, &_convFunc[0]);

    // Apply the image plane taper.
    _applyExpTaper(innerSize, _size, taperFactor, &_convFunc[0]);
}



void WProjConvFunc::generateUV(const unsigned innerSize, const unsigned padding,
        const float pixelSizeLM_rads, const float w,
        const float taperFactor, const float cutoff)
{
    // ==== Resize the function if needed.
    _size = innerSize * padding;
    if (_convFunc.size() != _size * _size)
        _convFunc.resize(_size * _size, Complex(0.0, 0.0));

    // ==== Generate the LM plane phase screen.
    _wFuncLMPadded(innerSize, _size, pixelSizeLM_rads, w, &_convFunc[0]);

    // ==== Apply the image plane taper function.
    _applyExpTaper(innerSize, _size, taperFactor, &_convFunc[0]);

    // ==== FFT to UV plane.
    _cfft2d(_size, &_convFunc[0]);

    // ==== Normalise
    float max = _findMax(_size, &_convFunc[0]);
    _scale(_size * _size, &_convFunc[0], 1.0f / max);

    // ==== Find the index at the cutoff level.
    // Search along the centre row going outwards from the centre.
    int cutoffIndex = -1;
    for (unsigned i = _size/2; i < _size; ++i)
    {
        const int idx = (_size / 2) * _size + i;
//        printf("%d %f\n", i - _size/2, abs(_convFunc[idx]));
        if (abs(_convFunc[idx]) < cutoff)
        {
            cutoffIndex = i - _size/2;
            break;
        }
    }
    printf("= cutoff index = %d\n", cutoffIndex);

    // ==== Convert cutoff level index to grid pixel space.
    // TODO: better way to do this using only half the size?
    // Maximum possible number of pixels in the convolution function.
    unsigned maxPixels = (unsigned) floor((float)_size / (float) padding);
    // Convolution functions need to be odd sized.
    if (maxPixels % 2 == 0) maxPixels--;
    const unsigned maxRadius = (maxPixels - 1) / 2;

    // Cutoff radius in grid pixels.
    int cutoffPixelRadius = 0;

    // Catch for when cutoff index isn't found.
    if (cutoffIndex == -1)
    {
        fprintf(stderr, "WProjConvFunc::generateUV(): Cutoff level not found!");
        cutoffPixelRadius = 1;
    }

    // Everything is good.
    else
    {
        cutoffPixelRadius = (int) ceil((float)cutoffIndex / (float) padding);
    }

    // Set upper limit to radius. TODO(should this ever happen?)
    if (cutoffPixelRadius > (int)maxRadius)
        cutoffPixelRadius = maxRadius;

    // Set the minimum number of pixels.
    if (cutoffPixelRadius < 2)
    {
        fprintf(stderr, "WProjConvFunc::generateUV(): "
                "Cutoff radius too small, defaulting to minimum size of 2.");
        cutoffPixelRadius = 2;
    }

    // ====  Reshape to a number of grid pixels.
    // TODO!
}



void WProjConvFunc::_wFuncLMPadded(const unsigned innerSize, const unsigned size,
        const float pixelSizeLM_rads, const float w, Complex * convFunc)
{
    const unsigned centre = (unsigned)ceil((float)size / 2.0f);
    const int radius = innerSize / 2.0f;
    const float twoPiW = M_PI * 2.0 * w;
    const float max_r2 = (radius * pixelSizeLM_rads) * (radius * pixelSizeLM_rads);

    for (int j = -radius; j <= radius; ++j)
    {
        for (int i = -radius; i <= radius; ++i)
        {
            const int idx = ((j + centre) * size) + centre + i;
            assert(idx < (int)size * size);

            const float m = (float)j * pixelSizeLM_rads;
            const float l = (float)i * pixelSizeLM_rads;
            const float r2 = (l * l) + (m * m);
            assert(r2 < 1.0f);

            // TODO: mmm not sure if this is a good idea...
            if (r2 < max_r2)
            {
                const float phase = -twoPiW * (sqrt(1.0f - r2) - 1.0f);
                convFunc[idx] = Complex(cos(phase), sin(phase));
            }
        }
    }
}



void WProjConvFunc::_applyExpTaper(const unsigned innerSize, const unsigned size,
        const float taperFactor, Complex * convFunc)
{
    const unsigned centre = (unsigned)ceil((float)size / 2.0f);
    const int radius = innerSize / 2.0f;

    for (int j = -radius; j <= radius; ++j)
    {
        for (int i = -radius; i <= radius; ++i)
        {
            const int idx = ((j + centre) * size) + centre + i;
            assert(idx < (int)size * size);

            const float x = (float)i;
            const float y = (float)j;
            const float x2 = x * x;
            const float y2 = y * y;
            const float taper = exp(-(x2 + y2) * taperFactor);

            convFunc[idx] *= taper;
        }
    }
}

// TODO(optimisation): Don't recreate the fftw plane each time this is called.
void WProjConvFunc::_cfft2d(const unsigned size, Complex * convFunc)
{
    FFTUtility::fftPhase(size, size, convFunc);
    fftwf_complex * c = reinterpret_cast<fftwf_complex*>(convFunc);
    int sign = FFTW_FORWARD;
    unsigned flags = FFTW_ESTIMATE;
    fftwf_plan plan = fftwf_plan_dft_2d(size, size, c, c, sign, flags);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    FFTUtility::fftPhase(size, size, convFunc);
}

float WProjConvFunc::_findMax(const unsigned size, Complex * convFunc)
{
    // TODO(optimisation): Can just use the mid point?
    float convmax = -numeric_limits<float>::max();
    for (unsigned i = 0; i < _size * _size; ++i)
        convmax = max(convmax, abs(_convFunc[i]));
    return convmax;
}


void WProjConvFunc::_scale(const unsigned size, Complex * convFunc, const float value)
{
    for (unsigned i = 0; i < size; ++i)
        convFunc[i] *= value;
}

} // namespace oskar
