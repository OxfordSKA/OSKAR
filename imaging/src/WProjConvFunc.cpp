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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

namespace oskar {

WProjConvFunc::WProjConvFunc()
{
}

WProjConvFunc::~WProjConvFunc()
{
}


void WProjConvFunc::generateLM(const unsigned size, const unsigned sizeLM,
        const unsigned iStart, const float pixelSizeLM_rads,
        const float w, const float taperFactor)
{
    if (_convFunc.size() != size * size)
        _convFunc.resize(size * size);

    // Generate the LM plane phase screen.
    _wFuncLMPadded(size, sizeLM, iStart, pixelSizeLM_rads, w, &_convFunc[0]);

    // Apply the image plane taper.
    _applyExpTaper(size, sizeLM, iStart, taperFactor, &_convFunc[0]);
}



void WProjConvFunc::generateUV(const unsigned size, const unsigned sizeLM,
        const unsigned iStart, const float pixelSizeLM_rads,
        const float w, const float taperFactor, const float /*cutoff*/)
{
    if (_convFunc.size() != size * size)
        _convFunc.resize(size * size);

    // Generate the LM plane phase screen.
    _wFuncLMPadded(size, sizeLM, iStart, pixelSizeLM_rads, w, &_convFunc[0]);

    // Apply the image plane taper function.
    _applyExpTaper(size, sizeLM, iStart, taperFactor, &_convFunc[0]);

    // FFT to UV plane.
    _cfft2d(size, &_convFunc[0]);

    // Normalise

    // Reshape the function at the cutoff level.


}



void WProjConvFunc::_wFuncLMPadded(const unsigned size, const unsigned sizeLM,
        const unsigned iStart, const float pixelSizeLM_rads,
        const float w, Complex * convFunc)
{
    const unsigned iCentre = (unsigned) floor((float)sizeLM / 2.0f);
    const float twoPiW = M_PI * 2.0 * w;

//    const float r2_max = 2 * (iCentre * pixelSizeLM_rads) * (iCentre * pixelSizeLM_rads);

    // Note: needs to start at 1 to keep the function symmetric!.
    // This is needed not to mess up the FFT of this function.
#pragma omp parallel for
    for (unsigned j = 1; j < sizeLM; ++j)
    {
        for (unsigned i = 1; i < sizeLM; ++i)
        {
            const int idx = (j + iStart) * size + iStart + i;
            assert(idx < (int)size * size);

            const float m = ((float)j - iCentre) * pixelSizeLM_rads;
            const float l = ((float)i - iCentre) * pixelSizeLM_rads;
            const float r2 = (l * l) + (m * m);
//            printf("%f\n", r2);
            assert(r2 < 1.0f);

            const float phase = -twoPiW * (sqrt(1.0f - r2) - 1.0f);
            convFunc[idx] = Complex(cos(phase), sin(phase));
        }
    }
}



void WProjConvFunc::_applyExpTaper(const unsigned size, const unsigned sizeLM,
        const unsigned iStart, const float taperFactor, Complex * convFunc)
{
    const unsigned iCentre = (unsigned) floor((float)sizeLM / 2.0f);

    // Note: needs to start at 1 to keep the function symmetric!.
    // This is needed not to mess up the FFT of this function.
#pragma omp parallel for
    for (unsigned j = 1; j < sizeLM; ++j)
    {
        for (unsigned i = 1; i < sizeLM; ++i)
        {
            const int idx = (j + iStart) * size + iStart + i;
            assert(idx < (int)size * size);

            const float x = (float)i - iCentre;
            const float y = (float)j - iCentre;

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


} // namespace oskar
