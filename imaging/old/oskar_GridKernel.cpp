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

#include "imaging/oskar_GridKernel.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


//template <class T>
//void oskar_GridKernel<T>::pillbox(const unsigned support, const unsigned oversample)
//{
//    _oversample = oversample;
//    _support    = support;
//    _size       = (_support * 2 + 1) * _oversample;
//
//    if (_values.size() != _size)
//        _values.resize(size);
//
//    const int centre = _size / 2; // FIXME? Rounding?
//    const double inc = 1.0 / _oversample;
//
//    T* values = &_values[0];
//
//    for (int i = 0; i < (int)size; ++i)
//    {
//        const double x     = double(i - centre) * inc;
//        const double abs_x = std::fabs(x);
//        if (abs_x > 1.0)
//            values[i] = 0.0;
//        else if (std::fabs(abs_x - 1.0) < std::numeric_limits<T>::epsilon())
//            values[i] = 0.5;
//        else
//            values[i] = 1.0;
//    }
//
//}
//
//
//template <class T>
//void oskar_GridKernel<T>::exp(const unsigned support, const unsigned oversample)
//{
//    _oversample = oversample;
//    _support    = support;
//    _size       = (_support * 2 + 1) * _oversample;
//
//    if (_values.size() != _size)
//        _values.resize(size);
//
//    const double inc   = 1.0 / _oversample;
//    const int centre   = _size / 2;
//    const double x_max = 3.0;
//    const double p1    = 1.0;
//    const double p2    = 2.0;
//    T* values = &_values[0];
//
//    for (int i = 0; i < (int)size; ++i)
//    {
//        const double x     = double(i - centre) * inc;
//        const double abs_x = std::fabs(x);
//        const double x2    = std::pow(abs_x * p1, p2);
//        if (abs_x < x_max)
//            values[i] = std::exp(-x2);
//        else
//            values[i] = 0.0;
//    }
//}
//
//
//
//template <class T>
//void oskar_GridKernel<T>::sinc(const unsigned support, const unsigned oversample)
//{
//    _oversample = oversample;
//    _support    = support;
//    _size       = (_support * 2 + 1) * _oversample;
//
//    if (_values.size() != _size)
//        _values.resize(size);
//
//    const double inc   = 1.0 / _oversample;
//    const int centre   = _size / 2;
//    const double x_max = 3.0;
//    const double p1    = M_PI;
//    T* values = &_values[0];
//
//    for (int i = 0; i < (int)size; ++i)
//    {
//        const double x = double(i - centre) * inc;
//        const double abs_x = std::abs(x);
//
//        if (std::fabs(abs_x) < std::numeric_limits<T>::epsilon())
//            values[i] = 1.0;
//
//        else if (abs_x < x_max)
//        {
//            const double arg = p1 * abs_x;
//            values[i] = sin(arg) / arg;
//        }
//    }
//}
//
//
//template <class T>
//void oskar_GridKernel<T>::exp_sinc(const unsigned support, const unsigned oversample)
//{
//    _oversample = oversample;
//    _support    = support;
//    _size       = (_support * 2 + 1) * _oversample;
//
//    if (_values.size() != _size)
//        _values.resize(size);
//
//    const double inc   = 1.0 / _oversample;
//    const int centre   = _size / 2;
//    const double x_max = 3.0;
//    const double p1    = M_PI / 1.55;
//    const double p2    = 1.0 / 2.52;
//    const double p3    = 2.0;
//    T* values = &_values[0];
//
//    for (int i = 0; i < (int)size; ++i)
//    {
//        const double x = double(i - centre) * inc;
//        const double abs_x = std::fabs(x);
//
//        if (abs_x < inc)
//            values[i] = 1.0;
//
//        else if (abs_x < x_max)
//        {
//            const double arg = p1 * abs_x;
//            const double ampSinc = std::sin(arg) / arg;
//
//            const double ampExp = std::exp(-std::pow((abs_x * p2), p3));
//            values[i] = ampExp * ampSinc;
//        }
//    }
//}
//
//
//template <class T>
//void oskar_GridKernel<T>::create_grid_kernel_image(T* image) const
//{
//    T* values = &_values[0];
//    for (unsigned j = 0; j < _size; ++j)
//    {
//        for (unsigned i = 0; i < _size; ++i)
//        {
//            image[j * _size + i] = values[j] * values[i];
//        }
//    }
//
//    return image;
//}

