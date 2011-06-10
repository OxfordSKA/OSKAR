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
#include <cstring>

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
    wFuncLMPadded(innerSize, _size, pixelSizeLM_rads, w, &_convFunc[0]);

    // Apply the image plane taper.
    applyExpTaper(innerSize, _size, taperFactor, &_convFunc[0]);
}



void WProjConvFunc::generateUV(const unsigned innerSize, const unsigned padding,
        const float pixelSizeLM_rads, const float w,
        const float taperFactor, const float cutoff, const bool reorder)
{
    // ==== Resize the function if needed.
    _size = innerSize * padding;
    if (_convFunc.size() != _size * _size)
        _convFunc.resize(_size * _size, Complex(0.0, 0.0));

    // ==== Generate the LM plane phase screen.
    wFuncLMPadded(innerSize, _size, pixelSizeLM_rads, w, &_convFunc[0]);

    // ==== Apply the image plane taper function.
    applyExpTaper(innerSize, _size, taperFactor, &_convFunc[0]);

    // ==== FFT to UV plane.
    cfft2d(_size, &_convFunc[0]);

    // ==== Normalise
    float max = findMax(_size, &_convFunc[0]);
    scale(_size * _size, &_convFunc[0], 1.0f / max);

    // ==== Find the index into the kernel at the cutoff level.
    const int cutoffIndex = findCutoffIndex(_size, &_convFunc[0], cutoff);

    // ==== Convert cutoff level index to grid pixel space.
    const unsigned cutoffPixelRadius = evaluateCutoffPixelRadius(_size, padding,
            cutoffIndex);

    // ====  Reshape to a number of grid pixels.
    reshape(cutoffPixelRadius, padding);


    // ==== Reorder convolution function memory (to make loading in gridding more
    // efficient?)
    if (reorder)
        reorder_memory(padding, _size, &_convFunc[0]);
}



void WProjConvFunc::wFuncLMPadded(const unsigned innerSize, const unsigned size,
        const float pixelSizeLM_rads, const float w, Complex * convFunc) const
{
    const unsigned centre = size / 2;
    const int radius = innerSize / 2;
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


void WProjConvFunc::applyExpTaper(const unsigned innerSize, const unsigned size,
        const float taperFactor, Complex * convFunc) const
{
    const unsigned centre = size / 2;
    const int radius = innerSize / 2;

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

            //convFunc[idx] = Complex(1.0f, 0.0f);//taper;
            convFunc[idx] *= taper;
        }
    }
}


// TODO(optimisation): Don't recreate the fftw plane each time this is called.
void WProjConvFunc::cfft2d(const unsigned size, Complex * convFunc) const
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


float WProjConvFunc::findMax(const unsigned size, Complex * convFunc) const
{
    // TODO(optimisation): Can just use the mid point?
    float convmax = -numeric_limits<float>::max();
    for (unsigned i = 0; i < size * size; ++i)
        convmax = max(convmax, abs(convFunc[i]));
    return convmax;
}


void WProjConvFunc::scale(const unsigned size, Complex * convFunc,
        const float value) const
{
    for (unsigned i = 0; i < size; ++i)
        convFunc[i] *= value;
}


int WProjConvFunc::findCutoffIndex(const unsigned size,
        const Complex * convFunc, const float cutoff) const
{
    int cutoffIndex = -1;
    for (unsigned i = size/2; i < size; ++i)
    {
        const int idx = (size / 2) * size + i;
        if (abs(convFunc[idx]) < cutoff)
        {
            cutoffIndex = i - size/2;
            break;
        }
    }
    return cutoffIndex;
}


unsigned WProjConvFunc::evaluateCutoffPixelRadius(const unsigned size,
        const unsigned padding, const int cutoffIndex, const unsigned minRadius)
{
    // TODO(better way to do this using only half the size?)
    // Maximum possible number of pixels in the convolution function.
    unsigned maxPixels = (unsigned) floor((float)size / (float) padding);
    // Convolution functions need to be odd sized.
    if (maxPixels % 2 == 0) maxPixels--;
    const unsigned maxRadius = (maxPixels - 1) / 2;

    // Cutoff radius in grid pixels.
    unsigned cutoffPixelRadius = 0;

    // Catch for when cutoff index isn't found.
    if (cutoffIndex == -1)
    {
        fprintf(stderr, "WProjConvFunc::generateUV(): Cutoff level not found!\n");
        cutoffPixelRadius = 1;
    }

    // Everything is good.
    else
    {
        cutoffPixelRadius = (int) ceil((float)cutoffIndex / (float) padding);
    }

    // Set upper limit to radius. TODO(should this ever happen?)
    if (cutoffPixelRadius > maxRadius)
        cutoffPixelRadius = maxRadius;

    // Set the minimum number of pixels.
    if (cutoffPixelRadius < minRadius)
    {
        fprintf(stderr, "WProjConvFunc::generateUV(): "
                "Cutoff radius too small, defaulting to minimum size of 2.\n");
        cutoffPixelRadius = minRadius;
    }
    return cutoffPixelRadius;
}


void WProjConvFunc::reshape(const unsigned cutoffPixelRadius,
        const unsigned padding)
{
    const unsigned cut_size = (cutoffPixelRadius * 2 + 1) * padding;
    vector<Complex> temp(cut_size * cut_size);
    const size_t num_bytes_row = cut_size * sizeof(Complex);
    const unsigned start_index = (_size / 2) - (int)floor((float) cut_size / 2.0f);

    for (unsigned j = 0; j < cut_size; ++j)
    {
        const unsigned from_index = (j + start_index) * _size + start_index;
        const Complex * from = &_convFunc[from_index];
        Complex * to = &temp[j * cut_size];
        memcpy((void*)to, (const void*)from, num_bytes_row);
    }
    _convFunc.resize(cut_size * cut_size);
    _size = cut_size;
    memcpy((void*)&_convFunc[0], (const void*)&temp[0], num_bytes_row * cut_size);
}



void WProjConvFunc::reorder_memory(const unsigned padding, const unsigned size,
        Complex * convFunc)
{
    const unsigned size_pixels = size / padding;
    const unsigned num_pixels = size_pixels * size_pixels;
    vector<Complex> temp(size * size);
    Complex * t = &temp[0];
    for (unsigned j = 0; j < size_pixels; ++j)
    {
        for (unsigned q = 0; q < padding; ++q)
        {
            for (unsigned i = 0; i < size_pixels; ++i)
            {
                for (unsigned p = 0; p < padding; ++p)
                {
                    const unsigned from_idx = (j * padding + q) * size + (i * padding + p);
                    const unsigned to_idx = (q * padding + p) * num_pixels + j * size_pixels + i;
                    t[to_idx] = convFunc[from_idx];
//                  printf("%d -> %d\n", from_idx, to_idx);
                }
            }
        }
    }
    memcpy((void*)convFunc, (const void*)t, size * size * sizeof(Complex));
}


} // namespace oskar
