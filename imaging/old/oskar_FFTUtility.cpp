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

#include "imaging/oskar_FFTUtility.h"

#include <cstring>
#include <fftw3.h>

namespace oskar {

std::complex<float> * FFTUtility::fftPhase(const unsigned nx, const unsigned ny,
        std::complex<float>* data)
{
    for (unsigned j = 0; j < ny; ++j)
    {
        for (unsigned i = 0; i < nx; ++i)
        {
            if ( (i + j) % 2 )
            {
                const unsigned idx = j * nx + i;
                data[idx] = -data[idx];
            }
        }
    }
    return data;
}


fftwf_complex * FFTUtility::fftPhase(const unsigned nx, const unsigned ny,
        fftwf_complex * data)
{
    for (unsigned j = 0; j < ny; ++j)
    {
        for (unsigned i = 0; i < nx; ++i)
        {
            const unsigned idx = j * nx + i;
            const int f = (int)pow(-1.0, i + j);
            data[idx][0] *= f;
            data[idx][1] *= f;
//            if ( (i + j) % 2 )
//            {
//                const unsigned idx = j * nx + i;
//                data[idx][0] = -data[idx][0];
//                data[idx][1] = -data[idx][1];
//            }
        }
    }
    return data;
}

float * FFTUtility::fftPhase(const unsigned nx, const unsigned ny,
        float * data)
{
    for (unsigned j = 0; j < ny; ++j)
    {
        for (unsigned i = 0; i < nx; ++i)
        {
            if ( (i + j) % 2 )
            {
                const unsigned idx = j * nx + i;
                data[idx] = -data[idx];
            }
        }
    }
    return data;
}



float * FFTUtility::fft_c2r_2d(const unsigned size, const std::complex<float>* in,
        float* out)
{
    // Copy grid to half complex section.
    const unsigned csize = size / 2 + 1;
    const size_t num_bytes = size * csize * sizeof(fftwf_complex);
    const fftwf_complex * cin = reinterpret_cast<const fftwf_complex*>(in);
    fftwf_complex * hcin = (fftwf_complex*) fftwf_malloc(num_bytes);

    for (unsigned j = 0; j < size; ++j)
    {
        const fftwf_complex * to = &hcin[j * csize];
        const fftwf_complex * from = &cin[j * size];
        memcpy((void*)to, (const void*)from, csize * sizeof(fftwf_complex));
    }

    // FFT.
    FFTUtility::fftPhase(csize, size, hcin);
    unsigned int flags = FFTW_ESTIMATE;
    fftwf_plan plan = fftwf_plan_dft_c2r_2d(size, size, hcin, out, flags);

    fftwf_execute(plan);
    FFTUtility::fftPhase(size, size, out);

    fftwf_free(hcin);
    fftwf_destroy_plan(plan);

    return out;
}


} // namespace oskar
