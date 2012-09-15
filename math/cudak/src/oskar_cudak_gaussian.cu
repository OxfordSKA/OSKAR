/*
 * Copyright (c) 2012, The University of Oxford
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


#include "math/cudak/oskar_cudak_gaussian.h"


__global__
void oskar_cudak_gaussian_f(float2* z, int n, const float* x, const float* y,
        float std)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    float x_ = x[i];
    float y_ = y[i];

    float arg = (x_*x_ + y_*y_) / (2.0 * std * std);
    z[i].x = expf(-arg);
    z[i].y = 0.0;
}


__global__
void oskar_cudak_gaussian_d(double2* z, int n, const double* x, const double* y,
        double std)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    double x_ = x[i];
    double y_ = y[i];

    double arg = (x_*x_ + y_*y_) / (2.0 * std * std);
    z[i].x = exp(-arg);
    z[i].y = 0.0;
}


__global__
void oskar_cudak_gaussian_mf(float4c* z, int n, const float* x, const float* y,
        float std)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    float x_ = x[i];
    float y_ = y[i];

    float arg = (x_*x_ + y_*y_) / (2.0 * std * std);
    float value = expf(-arg);
    z[i].a.x = value;
    z[i].a.y = 0.0;
    z[i].b.x = 0.0;
    z[i].b.y = 0.0;
    z[i].c.x = 0.0;
    z[i].c.y = 0.0;
    z[i].d.x = value;
    z[i].d.y = 0.0;
}


__global__
void oskar_cudak_gaussian_md(double4c* z, int n, const double* x, const double* y,
        double std)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    double x_ = x[i];
    double y_ = y[i];

    double arg = (x_*x_ + y_*y_) / (2.0 * std * std);
    double value = exp(-arg);
    z[i].a.x = value;
    z[i].a.y = 0.0;
    z[i].b.x = 0.0;
    z[i].b.y = 0.0;
    z[i].c.x = 0.0;
    z[i].c.y = 0.0;
    z[i].d.x = value;
    z[i].d.y = 0.0;
}
