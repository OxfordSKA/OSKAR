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

#ifndef CONV_FUNC_H_
#define CONV_FUNC_H_

namespace oskar {

/**
 * @class ConvFunc
 *
 * @brief
 *
 * @details
 * TODO(1) This should be a 1D evaluation used in a separable way.
 *    i.e. C(x,y) = C(x) * C(y)
 * TODO(2) Sort out interface in terms of radius, over-sample e.t.c.
 */

class ConvFunc
{
    public:
        ConvFunc();
        ~ConvFunc();

    public:
//        static float pillbox();

        static float exp(const float r, const float sigma = 1.0f);

//        static float sinc();

        static float expSinc(const float r);

//        static float spherodial();

    public:
        // FIXME: remove 2d versions in favour of separable functions...
        static void exp2D(const unsigned support, const unsigned oversample,
                const float sigma, float * cFunc);
        static void expSinc2D(const unsigned support, const unsigned oversample,
                float * cFunc);

    private:
        static float _expSinc(const float x, const float y);
};



} // namespace oskar
#endif // CONV_FUNC_H_
