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

#ifndef OSKAR_MATH_GRIDDER_H_
#define OSKAR_MATH_GRIDDER_H_

#include <complex>
#include <cmath>

typedef std::complex<float> Complex;

namespace oskar {


float oskar_math_gridder1(unsigned n, const float * x, const float * y,
        const Complex * amp, unsigned cSupport, unsigned cOversample,
        const float * cFunc, unsigned gSize, float pixelSize,
        Complex * grid, float * gridSum);


template <typename T> T _roundHalfUp(const T& x)
{
    return std::floor(x + 0.5);
}

template <typename T> T _roundHalfUp0(const T & x)
{
    T result = _roundHalfUp(std::fabs(x));
    return (x < 0.0) ? -result:result;
}


template <typename T> T _roundHalfDown(const T & x)
{
    return std::ceil(x - 0.5);
}


// this would also work... faster?
// -----
// (x > 0.0) ? floor(x) + centre : floor(x) + centre + 1;
// else centre;
// ------
template <typename T> T _roundHalfDown0(const T & x)
{
    T result = _roundHalfDown(std::fabs(x));
    return (x < 0.0) ? -result : result;
}


} // namespace oskar
#endif // OSKAR_MATH_GRIDDER_H_
