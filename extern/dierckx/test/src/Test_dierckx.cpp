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

#include "extern/dierckx/test/Test_dierckx.h"
#include "utility/timer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
void bispev_(float tx[], int* nx, float ty[], int* ny, float c[],
        int* kx, int* ky, float x[], int* mx, float y[], int* my,
        float z[], float wrk[], int* lwrk, int iwrk[], int* kwrk, int* ier);

void regrid_(int* iopt, int* mx, float x[], int* my, float y[], float z[],
        float* xb, float* xe, float* yb, float* ye, int* kx, int* ky,
        float* s, int* nxest, int* nyest, int* nx, float tx[], int* ny,
        float ty[], float c[], float* fp, float wrk[], int* lwrk, int iwrk[],
        int* kwrk, int* ier);
}

using std::vector;

void Test_dierckx::test_method()
{
    // Set data dimensions.
    int size_x = 10;
    int size_y = 10;

    // Set the input data.
    vector<float> data(size_x * size_y);

    // Create the data axes.
    vector<float> x(size_x), y(size_y);
    for (int i = 0; i < size_x; ++i)
        x[i] = (float) i;
    for (int i = 0; i < size_y; ++i)
        y[i] = (float) i;

    // Set up the surface fitting parameters.
    int iopt = 0; // -1 = Specify least-squares spline.
    int kxy = 3; // Degree of spline (cubic).
    float noise = 5e-4; // Numerical noise on input data.

    // Checks.
    if (size_x <= kxy || size_y <= kxy)
    {
        CPPUNIT_FAIL("ERROR: Input grid dimensions too small. Aborting.");
        return;
    }

    // Set up the spline knots.
    int nx = 0; // Number of knots in x.
    int ny = 0; // Number of knots in y.
    int nxest = size_x + kxy + 1; // Maximum number of knots in x.
    int nyest = size_y + kxy + 1; // Maximum number of knots in y.
    vector<float> tx(nxest, 0.0); // Spline knots in x.
    vector<float> ty(nyest, 0.0); // Spline knots in y.
    vector<float> c((nxest-kxy-1)*(nyest-kxy-1)); // Output spline coefficients.
    float fp = 0.0; // Output sum of squared residuals of spline approximation.

    // Set up workspace.
    int u = size_y > nxest ? size_y : nxest;
    int lwrk = 4 + nxest * (size_y + 2 * kxy + 5) +
            nyest * (2 * kxy + 5) + size_x * (kxy + 1) +
            size_y * (kxy + 1) + u;
    vector<float> wrk(lwrk);
    int kwrk = 3 + size_x + size_y + nxest + nyest;
    vector<int> iwrk(kwrk);
    int ier = 0; // Output return code.

    TIMER_START
    int k = 0;
    // Set initial smoothing factor (ignored for iopt < 0).
    float s = size_x * size_y * pow(noise, 2.0);
    bool fail = false;
    do
    {
        // Generate knot positions *at grid points* if required.
        if (iopt < 0 || fail)
        {
            int i, k, stride = 1;
            for (k = 0, i = kxy - 1; i <= size_x - kxy + stride; i += stride, ++k)
                tx[k + kxy + 1] = x[i]; // Knot x positions.
            nx = k + 2 * kxy + 1;
            for (k = 0, i = kxy - 1; i <= size_y - kxy + stride; i += stride, ++k)
                ty[k + kxy + 1] = y[i]; // Knot y positions.
            ny = k + 2 * kxy + 1;
        }

        // Set iopt to 1 if this is at least the second of multiple passes.
        if (k > 0) iopt = 1;
        regrid_(&iopt, &size_x, &x[0], &size_y, &y[0], &data[0],
                &x[0], &x[size_x-1], &y[0], &y[size_y-1], &kxy, &kxy, &s,
                &nxest, &nyest, &nx, &tx[0], &ny, &ty[0], &c[0], &fp,
                &wrk[0], &lwrk, &iwrk[0], &kwrk, &ier);
        if (ier == 1)
        {
            CPPUNIT_FAIL("ERROR: Workspace overflow.");
            return;
        }
        else if (ier == 2)
        {
            fprintf(stderr, "ERROR: Impossible result! (s too small?)\n");
            fprintf(stderr, "### Reverting to single-shot fit.\n");
            fail = true;
            k = 0;
            iopt = -1;
            continue;
        }
        else if (ier == 3)
        {
            fprintf(stderr, "ERROR: Iteration limit. (s too small?)\n");
            fprintf(stderr, "### Reverting to single-shot fit.\n");
            fail = true;
            k = 0;
            iopt = -1;
            continue;
        }
        else if (ier == 10)
        {
            CPPUNIT_FAIL("ERROR: Invalid input arguments.");
            return;
        }
        fail = false;

        // Print knot positions.
        printf(" ## Pass %d has knots (nx,ny)=(%d,%d), s=%.4f, fp=%.4f\n",
                k+1, nx, ny, s, fp);
        printf("    x:\n");
        for (int j = 0; j < nx; ++j) printf(" %.3f", tx[j]); printf("\n");
        printf("    y:\n");
        for (int j = 0; j < ny; ++j) printf(" %.3f", ty[j]); printf("\n\n");

        // Reduce smoothness parameter.
        s = s / 1.2;

        // Increment counter.
        ++k;
    } while (fp / ((nx-2*(kxy+1)) * (ny-2*(kxy+1))) > pow(2.0 * noise, 2) &&
            k < 1000 && (iopt >= 0 || fail));
    TIMER_STOP("Finished precalculation");

    // Interpolate.
    int out_x = 701;
    int out_y = 501;
    vector<float> output(out_x * out_y);
    TIMER_START
    int one = 1;
    for (int j = 0, k = 0; j < out_y; ++j)
    {
        float py = y[size_y - 1] * float(j) / (out_y - 1);
        for (int i = 0; i < out_x; ++i, ++k)
        {
            float val;
            float px = x[size_x - 1] * float(i) / (out_x - 1);
            bispev_(&tx[0], &nx, &ty[0], &ny, &c[0], &kxy, &kxy,
                    &px, &one, &py, &one, &val,
                    &wrk[0], &lwrk, &iwrk[0], &kwrk, &ier);
            if (ier != 0)
            {
                CPPUNIT_FAIL("ERROR: Spline evaluation failed\n");
                return;
            }
            output[k] = val;
        }
    }
    TIMER_STOP("Finished interpolation (%d points)", out_x * out_y);

    // Write out the interpolated data.
    FILE* file = fopen("test.dat", "w");
    for (int j = 0, k = 0; j < out_y; ++j)
    {
        for (int i = 0; i < out_x; ++i, ++k)
        {
            fprintf(file, "%10.6f ", output[k]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}
