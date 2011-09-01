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

#ifndef FLOATING_POINT_COMPARE_H_
#define FLOATING_POINT_COMPARE_H_

/**
 * @file oskar_floating_point_compare.h
 */

#include <cmath>
#include <limits>


// Simple floating point comparison function.
template <typename T> bool approxEqual(T b, T a)
{
    return std::fabs(a - b) < std::numeric_limits<T>::epsilon();
}

// More accurate comparison function using a relative error of 1.0e-5
// (i.e. 99.999% accuracy)
template <typename T> bool isEqual(T a, T b)
{
    if (std::fabs(a - b) < std::numeric_limits<T>::epsilon())
        return true;

    T relativeError;
    if (std::fabs(b) > std::fabs(a))
        relativeError = std::fabs((a - b) / b);
    else
        relativeError = std::fabs((a - b) / a);

    if (relativeError <= (T)1.0e-5)
        return true;

    return false;
}

#endif // FLOATING_POINT_COMPARE_H_
