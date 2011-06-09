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
#include <iostream>

using namespace std;

namespace oskar {

float Gridder::oskar_math_gridder1(const unsigned n, const float * x,
        const float * y, const Complex * amp, const unsigned cSupport,
        unsigned cOversample, const float * cFunc, const unsigned gSize,
        const float pixelSize, Complex * grid, float * gridSum)
{
    *gridSum = 0.0f;

    if (x == 0 || y == 0 || amp == 0 || cFunc == 0 || grid == 0)
    {
        cerr << "ERROR: oskar_math_gridder1() Input data error." << endl;
        return *gridSum;
    }

    const unsigned gCentre = (unsigned) floor((float)gSize / 2.0f);
    //const unsigned gSizeX = (unsigned) ceil((float)gSize / 2.0) + 1;
    const unsigned cSize = cSupport * 2 + 1;
    const unsigned cCentre = (unsigned) floor((float)(cSize * cOversample) / 2.0f);
    //const float cRadius = (float)cSize / 2.0f;

    // Loop over data points and apply them to the grid.
    for (unsigned i = 0; i < n; ++i)
    {
        // Scale the input coordinates to grid space.
        float xScaled = x[i] / pixelSize;
        float yScaled = y[i] / pixelSize;

        // Round to the closest grid cell.
        const int xGrid = roundHalfUp0(xScaled);
        const int yGrid = roundHalfUp0(yScaled);

        // Index into the grid array.
        const unsigned ixGrid = xGrid + gCentre;
        const unsigned iyGrid = yGrid + gCentre;

        // Scaled distance from the nearest grid point.
        const float xOffset = (float)xGrid - xScaled;
        const float yOffset = (float)yGrid - yScaled;

        // Kernel offset.
//        const float xDelta = xOffset * (float)cOversample;
//        const float yDelta = yOffset * (float)cOversample;

        // Kernel offset.
        const int ixConvFunc = roundHalfDown0(xOffset) + cCentre;
        const int iyConvFunc = roundHalfDown0(yOffset) + cCentre;

//        cout << "---------------" << endl;
//        cout << "x[" << i << "] = " << x[i] << endl;
//        cout << "xScaled     = " << xScaled << endl;
//        cout << "xGrid       = " << xGrid << endl;
//        cout << "ixGrid      = " << ixGrid << endl;
//        cout << "xOffset     = " << xOffset << endl;
////        cout << "xDelta      = " << xDelta << endl;
//        cout << "ixConvFunc  = " << ixConvFunc << endl;

        for (unsigned y = 0; y < cSize; ++y)
        {
            for (unsigned x = 0; x < cSize; ++x)
            {
                const unsigned gx = ixGrid - cSupport + x;
                const unsigned gy = iyGrid - cSupport + y;
                const unsigned gIdx = gy * gSize + gx;
//                cout << "-- gx, gy = " << gx << ", " << gy << endl;


                const unsigned cy = iyConvFunc + (y - cSupport) * cOversample;
                const unsigned cx = ixConvFunc + (x - cSupport) * cOversample;
                const unsigned cIdx = cy * (cSize * cOversample) + cx;

                const float re = cFunc[cIdx] * amp[i].real();
                const float im = cFunc[cIdx] * amp[i].imag();
                grid[gIdx] += Complex(re, im);
                *gridSum += cFunc[cIdx];
            }
        }
    }

    return *gridSum;
}

} // namespace oskar

