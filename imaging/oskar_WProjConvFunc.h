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

#ifndef WPROJ_CONV_FUNC_H_
#define WPROJ_CONV_FUNC_H_

#include <vector>
#include <complex>

namespace oskar {

/**
 * @class WProjConvFunc
 *
 * @brief
 * W-projection convolution kernels.
 *
 * @details
 */

class WProjConvFunc
{
    public:
        typedef std::complex<float> Complex;

        WProjConvFunc();
        ~WProjConvFunc();

    public:
        Complex const * values() const { return  &_convFunc[0]; }

        unsigned size() const { return _size; }

    public:
        void generateLM(const unsigned innerSize, const unsigned padding,
                const float pixelSizeLM_rads,
                const float w, const float taperFactor);

        void generateUV(const unsigned innerSize, const unsigned padding,
                const float pixelSizeLM_rads, const float w,
                const float taperFactor, const float cutoff,
                const bool reorder = false);

    private:
        void wFuncLMPadded(const unsigned innerSize, const unsigned size,
                const float pixelSizeLM_rads, const float w,
                Complex * convFunc) const;

        void applyExpTaper(const unsigned innerSize, const unsigned size,
                const float taperFactor, Complex * convFunc) const;

        void cfft2d(const unsigned size, Complex * convFunc) const;

        float findMax(const unsigned size, Complex * convFunc) const;

        void scale(const unsigned size, Complex * convFunc, const float value) const;

        int findCutoffIndex(const unsigned size, const Complex * convFunc,
                const float cutoff) const;

        unsigned evaluateCutoffPixelRadius(const unsigned size,
                const unsigned padding, const int cutoffIndex,
                const unsigned minRadius = 2);

        void reshape(const unsigned cutoffPixelRadius, const unsigned padding);

        void reorder_memory(const unsigned padding, const unsigned size,
                Complex * convFunc);

    private:
        std::vector<Complex> _convFunc;
        unsigned _size;
};



} // namespace oskar
#endif // WPROJ_CONV_FUNC_H_
