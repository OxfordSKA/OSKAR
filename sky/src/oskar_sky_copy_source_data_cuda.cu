/*
 * Copyright (c) 2014, The University of Oxford
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

#include <oskar_sky_copy_source_data_cuda.h>
#include <thrust/device_vector.h> // Must be included before thrust/copy.h
#include <thrust/copy.h>

using thrust::copy_if;
using thrust::device_pointer_cast;
using thrust::device_ptr;

struct is_true
{
    __host__ __device__
    bool operator()(const int x) {return (bool)x;}
};

#define DPC(ptr) device_pointer_cast(ptr)

// Don't pass structure pointers here, because this takes ages to compile,
// and we don't want to keep recompiling if any of the other headers change!

extern "C"
void oskar_sky_copy_source_data_cuda_f(
        int num, int* num_out, const int* mask,
        const float* ra_in,   float* ra_out,
        const float* dec_in,  float* dec_out,
        const float* I_in,    float* I_out,
        const float* Q_in,    float* Q_out,
        const float* U_in,    float* U_out,
        const float* V_in,    float* V_out,
        const float* ref_in,  float* ref_out,
        const float* sp_in,   float* sp_out,
        const float* rm_in,   float* rm_out,
        const float* l_in,    float* l_out,
        const float* m_in,    float* m_out,
        const float* n_in,    float* n_out,
        const float* a_in,    float* a_out,
        const float* b_in,    float* b_out,
        const float* c_in,    float* c_out,
        const float* maj_in,  float* maj_out,
        const float* min_in,  float* min_out,
        const float* pa_in,   float* pa_out
        )
{
    device_ptr<const int> m = device_pointer_cast(mask);

    // Copy sources to new arrays based on mask values.
    device_ptr<float> out = copy_if(DPC(ra_in), DPC(ra_in) + num, m,
            DPC(ra_out), is_true());
    copy_if(DPC(dec_in), DPC(dec_in) + num, m, DPC(dec_out), is_true());
    copy_if(DPC(I_in), DPC(I_in) + num, m, DPC(I_out), is_true());
    copy_if(DPC(Q_in), DPC(Q_in) + num, m, DPC(Q_out), is_true());
    copy_if(DPC(U_in), DPC(U_in) + num, m, DPC(U_out), is_true());
    copy_if(DPC(V_in), DPC(V_in) + num, m, DPC(V_out), is_true());
    copy_if(DPC(ref_in), DPC(ref_in) + num, m, DPC(ref_out), is_true());
    copy_if(DPC(sp_in), DPC(sp_in) + num, m, DPC(sp_out), is_true());
    copy_if(DPC(rm_in), DPC(rm_in) + num, m, DPC(rm_out), is_true());
    copy_if(DPC(l_in), DPC(l_in) + num, m, DPC(l_out), is_true());
    copy_if(DPC(m_in), DPC(m_in) + num, m, DPC(m_out), is_true());
    copy_if(DPC(n_in), DPC(n_in) + num, m, DPC(n_out), is_true());
    copy_if(DPC(a_in), DPC(a_in) + num, m, DPC(a_out), is_true());
    copy_if(DPC(b_in), DPC(b_in) + num, m, DPC(b_out), is_true());
    copy_if(DPC(c_in), DPC(c_in) + num, m, DPC(c_out), is_true());
    copy_if(DPC(maj_in), DPC(maj_in) + num, m, DPC(maj_out), is_true());
    copy_if(DPC(min_in), DPC(min_in) + num, m, DPC(min_out), is_true());
    copy_if(DPC(pa_in), DPC(pa_in) + num, m, DPC(pa_out), is_true());

    // Get the number of sources copied.
    *num_out = out - DPC(ra_out);
}

extern "C"
void oskar_sky_copy_source_data_cuda_d(
        int num, int* num_out, const int* mask,
        const double* ra_in,   double* ra_out,
        const double* dec_in,  double* dec_out,
        const double* I_in,    double* I_out,
        const double* Q_in,    double* Q_out,
        const double* U_in,    double* U_out,
        const double* V_in,    double* V_out,
        const double* ref_in,  double* ref_out,
        const double* sp_in,   double* sp_out,
        const double* rm_in,   double* rm_out,
        const double* l_in,    double* l_out,
        const double* m_in,    double* m_out,
        const double* n_in,    double* n_out,
        const double* a_in,    double* a_out,
        const double* b_in,    double* b_out,
        const double* c_in,    double* c_out,
        const double* maj_in,  double* maj_out,
        const double* min_in,  double* min_out,
        const double* pa_in,   double* pa_out
        )
{
    device_ptr<const int> m = device_pointer_cast(mask);

    // Copy sources to new arrays based on mask values.
    device_ptr<double> out = copy_if(DPC(ra_in), DPC(ra_in) + num, m,
            DPC(ra_out), is_true());
    copy_if(DPC(dec_in), DPC(dec_in) + num, m, DPC(dec_out), is_true());
    copy_if(DPC(I_in), DPC(I_in) + num, m, DPC(I_out), is_true());
    copy_if(DPC(Q_in), DPC(Q_in) + num, m, DPC(Q_out), is_true());
    copy_if(DPC(U_in), DPC(U_in) + num, m, DPC(U_out), is_true());
    copy_if(DPC(V_in), DPC(V_in) + num, m, DPC(V_out), is_true());
    copy_if(DPC(ref_in), DPC(ref_in) + num, m, DPC(ref_out), is_true());
    copy_if(DPC(sp_in), DPC(sp_in) + num, m, DPC(sp_out), is_true());
    copy_if(DPC(rm_in), DPC(rm_in) + num, m, DPC(rm_out), is_true());
    copy_if(DPC(l_in), DPC(l_in) + num, m, DPC(l_out), is_true());
    copy_if(DPC(m_in), DPC(m_in) + num, m, DPC(m_out), is_true());
    copy_if(DPC(n_in), DPC(n_in) + num, m, DPC(n_out), is_true());
    copy_if(DPC(a_in), DPC(a_in) + num, m, DPC(a_out), is_true());
    copy_if(DPC(b_in), DPC(b_in) + num, m, DPC(b_out), is_true());
    copy_if(DPC(c_in), DPC(c_in) + num, m, DPC(c_out), is_true());
    copy_if(DPC(maj_in), DPC(maj_in) + num, m, DPC(maj_out), is_true());
    copy_if(DPC(min_in), DPC(min_in) + num, m, DPC(min_out), is_true());
    copy_if(DPC(pa_in), DPC(pa_in) + num, m, DPC(pa_out), is_true());

    // Get the number of sources copied.
    *num_out = out - DPC(ra_out);
}
