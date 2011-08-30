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
