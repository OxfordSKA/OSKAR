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

#include "imaging/oskar_evaluate_gridding_kernels.h"
#include "stdlib.h"
#include "math.h"
#include "float.h"
#include "string.h" // for memset
#include "stdio.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef MAX
#define MAX( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef MIN
#define MIN( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif


#ifdef __cplusplus
extern "C" {
#endif


void oskar_initialise_kernel_d(const double radius, oskar_GridKernel_d* kernel)
{
    kernel->radius     = radius;
    kernel->num_cells  = MAX(kernel->radius, 1.0) * 2 + 1;
    kernel->oversample = 100;
    kernel->oversample = (kernel->oversample / 2) * 2;
    kernel->size       = kernel->num_cells * kernel->oversample + 1;
    kernel->centre     = kernel->oversample / 2 * kernel->num_cells;
    kernel->xinc       = 1.0 / kernel->oversample;
    kernel->amp     = (double*) malloc(kernel->size * sizeof(double));
//    printf("radius     = %f\n", kernel->radius);
//    printf("num cells  = %i\n", kernel->num_cells);
//    printf("oversample = %i\n", kernel->oversample);
//    printf("size       = %i\n", kernel->size);
//    printf("centre     = %i\n", kernel->centre);
//    printf("num cells  = %i\n", kernel->num_cells);
//    printf("xinc       = %f\n", kernel->xinc);
}


/*
 * Based on AIPS convolution function type 1 (HELP UV1TYPE)
 * PARAM(1) -> Support size in cell 0.0 --> 0.5;
 *
 */
void oskar_evaluate_pillbox_d(oskar_GridKernel_d* kernel)
{
    oskar_initialise_kernel_d(0.5, kernel);

    for (int i = 0; i < (int)kernel->size; ++i)
    {
        const double x    = (i - (int)kernel->centre) * kernel->xinc;
        const double absx = fabs(x);
        if (fabs(absx - kernel->radius) < DBL_EPSILON)
            kernel->amp[i] = 0.5;
        else if (absx > kernel->radius)
            kernel->amp[i] = 0.0;
        else
            kernel->amp[i] = 1.0;
    }
}


void oskar_evaluate_exp_sinc_d(oskar_GridKernel_d* kernel)
{
    oskar_initialise_kernel_d(3.0, kernel);

    const double p1 = M_PI / 1.55;
    const double p2 = 1.0 / 2.52;
    const double p3 = 2.0;

    for (int i = 0; i < (int)kernel->size; ++i)
    {
        const double x    = (i - (int)kernel->size /2) * kernel->xinc;
        const double absx = fabs(x);

        kernel->amp[i] = 0.0;

        if (absx < kernel->xinc)
            kernel->amp[i] = 1.0;

        else if (absx < kernel->radius)
            kernel->amp[i] = sin(x * p1) / (x * p1) * exp(-pow(absx * p2, p3));
    }
}



void oskar_evaluate_spheroidal_d(oskar_GridKernel_d* kernel)
{
    // AIPS: HELP UV5TYPE
    // param1 => Support size. 0.0 -> 3.0 cells.
    // param2 => ALPHA. 0.0 -> 1.0.
    // param3 => Scale factor for support size. 0.0 -> 1.0.

    double param1 = 2.0; // related to support radius.
    //double param2 = 0.0; // related to alpha (0.0 = recommended?)

    oskar_initialise_kernel_d(param1, kernel);
    memset(kernel->amp, 0, kernel->size * sizeof(double));

    // TODO: check these!
    int iAlpha = 1;
    // alpha[iAlpha] = {0, 1/2, 1, 3/2, 2}
    //int iAlpha = MAX(0, MIN(4, 2.0 * param2 + 1.1));
    //int im = MAX(3, MIN(7, 2.0 * param1 + 0.1));
    //int im = 5;
//    printf("iAlpha = %i, im = %i\n", iAlpha, im);

    int nmax = kernel->radius / kernel->xinc + 0.1;
    //int nmax = kernel->centre;
//    printf("nmax = %i\n", nmax);

    // Evaluate function.
    for (int i = 0; i < nmax; ++i)
    {
        const double eta = i / (double)(nmax - 1);
        double value = 0.0;
        spheroidal_d(iAlpha, 0, eta, &value);
        const int index = kernel->centre + i;
        kernel->amp[index] = value;
    }

    // Fill in the other half.
    for (int i = 0; i < (int)kernel->centre-1; ++i)
        kernel->amp[kernel->centre - i] = kernel->amp[kernel->centre + i];
}




void spheroidal_d(int iAlpha, int iflag, double eta, double* value)
{
    // Look-up table (taken from AIPS SPHFN.FOR)
    double p5[9][4] =
    {
        {  3.722238E-3, -4.991683E-2,  1.658905E-1, -2.387240E-1 },
        {  1.877469E-1, -8.159855E-2,  3.051959E-2,  8.182649E-3 },
        { -7.325459E-2,  1.945697E-1, -2.396387E-1,  1.667832E-1 },
        { -6.620786E-2,  2.224041E-2,  1.466325E-2, -9.858686E-2 },
        {  2.180684E-1, -2.347118E-1,  1.464354E-1, -5.350728E-2 },
        {  1.624782E-2,  2.314317E-2, -1.246383E-1,  2.362036E-1 },
        {  2.257366E-1,  1.275895E-1, -4.317874E-2,  1.193168E-2 },
        {  3.346886E-2, -1.503778E-1,  2.492826E-1, -2.142055E-1 },
        {  1.106482E-1, -3.486024E-2,  8.821107E-3,  NAN         }
    };

    double q5[5] =
    { 2.418820E-1,  2.291233E-1,  2.177793E-1,  2.075784E-1, 1.983358E-1 };

    double alpha[5] = { 0.0, 0.5, 1.0, 1.5, 2.0 };

    // Check inputs.
    if (iAlpha < 0 || iAlpha > 4)
    {
        fprintf(stderr, "iAlpha out of range!\n");
        *value = -99.0;
        return;
    }
    if (fabs(eta) > 1.0)
    {
        fprintf(stderr, "eta out of range!\n");
        *value = -99.0;
        return;
    }

    double eta2 = pow(eta, 2.0);
    int j = iAlpha;

    *value = 0.0;
    if (fabs(eta) == 1.0)
        return;

    double x = eta2 - 1.0;

    *value = (p5[0][j] + x * (p5[1][j] + x * (p5[2][j] + x * (p5[3][j] +
            x * (p5[4][j] + x * (p5[5][j] + x * p5[6][j] ))))))
            / (1.0 + x * q5[j]);

    if (iflag > 0 || iAlpha == 0 || eta == 0.0)
        return;

    *value *= pow(1.0 - eta2, alpha[iAlpha]);
}


#ifdef __cplusplus
}
#endif
