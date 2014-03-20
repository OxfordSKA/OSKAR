/*
 * Copyright (c) 2013-2014, The University of Oxford
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

#include <oskar_sky_filter_by_flux_cuda.h>

#include <thrust/device_vector.h>
#include <thrust/remove.h>

using thrust::remove_if;
using thrust::device_pointer_cast;
using thrust::device_ptr;

template<typename T>
struct is_outside_range
{
    __host__ __device__
    bool operator()(const T x) { return !(x > min_f && x <= max_f); }
    T min_f;
    T max_f;
};

#define DPC(ptr) device_pointer_cast(ptr)

// Don't pass structure pointers here, because this takes ages to compile,
// and we don't want to keep recompiling if any of the other headers change!

extern "C"
void oskar_sky_filter_by_flux_cuda_f(int num, int* num_out, float min_I,
        float max_I, float* ra, float* dec, float* I, float* Q, float* U,
        float* V, float* ref, float* sp, float* rm, float* l, float* m,
        float* n, float* a, float* b, float* c, float* maj, float* min,
        float* pa)
{
    is_outside_range<float> range_check;
    device_ptr<const float> Ic = DPC((const float*) I);
    range_check.min_f = min_I;
    range_check.max_f = max_I;

    // Remove sources outside Stokes I range.
    device_ptr<float> out = remove_if(DPC(ra), DPC(ra) + num, Ic, range_check);
    remove_if(DPC(dec), DPC(dec) + num, Ic, range_check);
    remove_if(DPC(Q), DPC(Q) + num, Ic, range_check);
    remove_if(DPC(U), DPC(U) + num, Ic, range_check);
    remove_if(DPC(V), DPC(V) + num, Ic, range_check);
    remove_if(DPC(ref), DPC(ref) + num, Ic, range_check);
    remove_if(DPC(sp), DPC(sp) + num, Ic, range_check);
    remove_if(DPC(rm), DPC(rm) + num, Ic, range_check);
    remove_if(DPC(l), DPC(l) + num, Ic, range_check);
    remove_if(DPC(m), DPC(m) + num, Ic, range_check);
    remove_if(DPC(n), DPC(n) + num, Ic, range_check);
    remove_if(DPC(a), DPC(a) + num, Ic, range_check);
    remove_if(DPC(b), DPC(b) + num, Ic, range_check);
    remove_if(DPC(c), DPC(c) + num, Ic, range_check);
    remove_if(DPC(maj), DPC(maj) + num, Ic, range_check);
    remove_if(DPC(min), DPC(min) + num, Ic, range_check);
    remove_if(DPC(pa), DPC(pa) + num, Ic, range_check);

    // Finally, remove Stokes I values.
    remove_if(DPC(I), DPC(I) + num, DPC(I), range_check);

    *num_out = out - DPC(ra);
}

extern "C"
void oskar_sky_filter_by_flux_cuda_d(int num, int* num_out, double min_I,
        double max_I, double* ra, double* dec, double* I, double* Q, double* U,
        double* V, double* ref, double* sp, double* rm, double* l, double* m,
        double* n, double* a, double* b, double* c, double* maj, double* min,
        double* pa)
{
    is_outside_range<double> range_check;
    device_ptr<const double> Ic = DPC((const double*) I);
    range_check.min_f = min_I;
    range_check.max_f = max_I;

    // Remove sources outside Stokes I range.
    device_ptr<double> out = remove_if(DPC(ra), DPC(ra) + num, Ic, range_check);
    remove_if(DPC(dec), DPC(dec) + num, Ic, range_check);
    remove_if(DPC(Q), DPC(Q) + num, Ic, range_check);
    remove_if(DPC(U), DPC(U) + num, Ic, range_check);
    remove_if(DPC(V), DPC(V) + num, Ic, range_check);
    remove_if(DPC(ref), DPC(ref) + num, Ic, range_check);
    remove_if(DPC(sp), DPC(sp) + num, Ic, range_check);
    remove_if(DPC(rm), DPC(rm) + num, Ic, range_check);
    remove_if(DPC(l), DPC(l) + num, Ic, range_check);
    remove_if(DPC(m), DPC(m) + num, Ic, range_check);
    remove_if(DPC(n), DPC(n) + num, Ic, range_check);
    remove_if(DPC(a), DPC(a) + num, Ic, range_check);
    remove_if(DPC(b), DPC(b) + num, Ic, range_check);
    remove_if(DPC(c), DPC(c) + num, Ic, range_check);
    remove_if(DPC(maj), DPC(maj) + num, Ic, range_check);
    remove_if(DPC(min), DPC(min) + num, Ic, range_check);
    remove_if(DPC(pa), DPC(pa) + num, Ic, range_check);

    // Finally, remove Stokes I values.
    remove_if(DPC(I), DPC(I) + num, DPC(I), range_check);

    *num_out = out - DPC(ra);
}

