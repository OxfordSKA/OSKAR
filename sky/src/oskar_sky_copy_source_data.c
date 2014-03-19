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

#include <oskar_sky_copy_source_data.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_copy_source_data_f(
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
    int i, i_out = 0;
    for (i = 0; i < num; ++i)
    {
        if (mask[i])
        {
            ra_out[i_out]  = ra_in[i];
            dec_out[i_out] = dec_in[i];
            I_out[i_out]   = I_in[i];
            Q_out[i_out]   = Q_in[i];
            U_out[i_out]   = U_in[i];
            V_out[i_out]   = V_in[i];
            ref_out[i_out] = ref_in[i];
            sp_out[i_out]  = sp_in[i];
            rm_out[i_out]  = rm_in[i];
            l_out[i_out]   = l_in[i];
            m_out[i_out]   = m_in[i];
            n_out[i_out]   = n_in[i];
            a_out[i_out]   = a_in[i];
            b_out[i_out]   = b_in[i];
            c_out[i_out]   = c_in[i];
            maj_out[i_out] = maj_in[i];
            min_out[i_out] = min_in[i];
            pa_out[i_out]  = pa_in[i];
            (i_out)++;
        }
    }
    *num_out = i_out;
}

void oskar_sky_copy_source_data_d(
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
    int i, i_out = 0;
    for (i = 0; i < num; ++i)
    {
        if (mask[i])
        {
            ra_out[i_out]  = ra_in[i];
            dec_out[i_out] = dec_in[i];
            I_out[i_out]   = I_in[i];
            Q_out[i_out]   = Q_in[i];
            U_out[i_out]   = U_in[i];
            V_out[i_out]   = V_in[i];
            ref_out[i_out] = ref_in[i];
            sp_out[i_out]  = sp_in[i];
            rm_out[i_out]  = rm_in[i];
            l_out[i_out]   = l_in[i];
            m_out[i_out]   = m_in[i];
            n_out[i_out]   = n_in[i];
            a_out[i_out]   = a_in[i];
            b_out[i_out]   = b_in[i];
            c_out[i_out]   = c_in[i];
            maj_out[i_out] = maj_in[i];
            min_out[i_out] = min_in[i];
            pa_out[i_out]  = pa_in[i];
            i_out++;
        }
    }
    *num_out = i_out;
}

#ifdef __cplusplus
}
#endif
