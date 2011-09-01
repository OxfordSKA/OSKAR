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

#ifndef OSKAR_GRID_KERNEL_H_
#define OSKAR_GRID_KERNEL_H_

#include "utility/oskar_vector_types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_GridKernel_d
{
    double  radius;
    int     num_cells;
    int     oversample;
    int     size;
    int     centre;
    double  xinc;
    double* values;
};
typedef struct oskar_GridKernel_d oskar_GridKernel_d;



struct oskar_WProjGridKernel_d
{
    double   radius;
    int      num_cells;
    int      oversample;
    int      size;
    int      centre;
    double   xinc;
    double2* values;
};
typedef struct oskar_WProjGridKernel_d oskar_WProjGridKernel_d;


#ifdef __cplusplus
}
#endif


//
//
//#include <vector>
//
///**
// * @class oskar_GridKernel
// *
// * @brief
// *
// * @details
// */
//
//template <class T>
//class oskar_GridKernel
//{
//    public:
//        /// Constructor.
//        oskar_GridKernel() : _size(0), _support(0), _oversample(0) {}
//
//        /// Destructor.
//        ~oskar_GridKernel() {}
//
//    public:
//        /// Return a pointer to the gridding kernel data.
//        const T* values() const { return  &_values[0]; }
//
//        /// Return the size of the gridding kernel.
//        unsigned size() const { return _size; }
//
//        /// Return the support radius of the gridding kernel.
//        unsigned support() const { return _support; }
//
//        /// Return the number of pixels per cell (oversample).
//        unsigned oversample() const { return _oversample; }
//
//    public:
//        /// Generate a pill-box gridding kernel.
//        void pillbox(const unsigned support, const unsigned oversample);
//
//        /// Generate a Gaussian gridding kernel.
//        void exp(const unsigned support, const unsigned oversample);
//
//        /// Generate a Sinc gridding kernel.
//        void sinc(const unsigned support, const unsigned oversample);
//
//        /// Generate a Gaussian times Sinc gridding kernel.
//        void exp_sinc(const unsigned support, const unsigned oversample);
//
//    public:
//        // NOTE(For testing only)
//        // WARNING: assumes image has already been declare to size:
//        //      _size * _size * sizeof(T)
//        void create_grid_kernel_image(T* image) const;
//
//    private:
//        unsigned _size;
//        unsigned _support;
//        unsigned _oversample;
//        std::vector<T> _values;
//};



#endif // OSKAR_GRID_KERNEL_H_
