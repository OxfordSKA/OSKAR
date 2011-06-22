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

#include "imaging/Gridder.h"
#include "math/core/Rounding.h"

#include <cstdio>
#include <cmath>

using namespace std;

namespace oskar {

void Gridder::grid_standard(
        const unsigned num_data,
        const float * data_x,
        const float * data_y,
        const Complex * data_amp,
        const unsigned support,
        const unsigned oversample,
        const float * conv_func,
        const unsigned grid_size,
        const float pixel_size,
        Complex * grid,
        double * grid_sum)
{
    *grid_sum = 0.0;

    const unsigned gcentre = (unsigned) floor((float)grid_size / 2.0f);
    const unsigned csize = (support * 2) + 1;
    const unsigned ccentre = (unsigned) floor((float)(csize * oversample) / 2.0f);

    int x_grid, x_conv_func, y_grid, y_conv_func;

    // Loop over data points and apply them to the grid.
    for (unsigned i = 0; i < num_data; ++i)
    {
        calculate_offset(data_x[i], pixel_size, oversample, &x_grid, &x_conv_func);
        calculate_offset(data_y[i], pixel_size, oversample, &y_grid, &y_conv_func);

        x_grid += gcentre;
        y_grid += gcentre;
        x_conv_func += ccentre;
        y_conv_func += ccentre;

        const float aRe = data_amp[i].real();
        const float aIm = data_amp[i].imag();

        for (int iy = -(int)support; iy <= (int)support; ++iy)
        {
            for (int ix = -(int)support; ix <= (int)support; ++ix)
            {
                const int cx = (ix * (int)oversample) + x_conv_func;
                const int cy = (iy * (int)oversample) + y_conv_func;
                const float c = conv_func[cx] * conv_func[cy];

                const int gx = x_grid + ix;
                const int gy = y_grid + iy;

                grid[gy * grid_size + gx] += Complex(c * aRe, c * aIm);

                *grid_sum += (double)c;
            }
        }
    }
}


void Gridder::degrid_standard()
{
}



void Gridder::calculate_offset(const float x, const float pixel_size,
        const unsigned oversample, int * x_grid,
        int * x_conv_func)
{
    // Scale the input coordinates to grid space.
    const float xScaled = x / pixel_size;

    // Evaluate the closest grid cell.
//    *x_grid = int(xScaled);
    *x_grid = (int)round_towards_zero(xScaled);
    //*x_grid = (int)round_away_from_zero(xScaled);

    // Float distance from data point to nearest grid cell.
    const float grid_delta = (*x_grid - xScaled);

    // Increment in the convolution function look-up table.
    const float conv_inc = 1.0f / static_cast<float>(oversample);

    // Evaluate the index of the convolution kernel at the closest grid cell.
    const float conv_delta = grid_delta / conv_inc;
    //*x_conv_func = int(conv_delta);
    *x_conv_func = (int)round_towards_zero(conv_delta);
    //*x_conv_func = (int)round_away_from_zero(conv_delta);

//    printf("conv_inc   = %f\n", conv_inc);
//    printf("grid_delta = %f\n", grid_delta);
//    printf("conv_delta = %f\n", conv_delta);
//    printf("conv_x     = %d\n", *x_conv_func);
}




float Gridder::round_away_from_zero(const float x)
{
    return (x > 0.0f) ? std::floor(x + 0.5) : std::ceil(x - 0.5);
}


float Gridder::round_towards_zero(const float x)
{
    return (x > 0.0f) ? std::ceil(x - 0.5) : std::floor(x + 0.5);
}


} // namespace oskar

