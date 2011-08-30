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

namespace oskar {

class Gridder
{
    public:
        typedef std::complex<float> Complex;

        // Standard gridding.
        void grid_standard(const unsigned num_data,
                const float * data_x,
                const float * data_y,
                const Complex * data_amp,
                const unsigned support,
                const unsigned oversample,
                const float * conv_func,
                const unsigned grid_size,
                const float pixel_size,
                Complex * grid,
                double * grid_sum);

        // Standard degridding.
        void degrid_standard();

        // WProjection gridding.


        // WProjection degridding.

    public:
        void calculate_offset(const float x, const float pixel_size,
                const unsigned oversample, int * x_grid,
                int * x_conv_func);

    private:
        // Round away from zero (symmetric about zero).
        float round_away_from_zero(const float x);

        // Round towards from zero (symmetric about zero).
        float round_towards_zero(const float x);
};


} // namespace oskar
#endif // OSKAR_MATH_GRIDDER_H_
