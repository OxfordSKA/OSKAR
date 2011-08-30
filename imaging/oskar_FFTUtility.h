#ifndef FFT_UTILITY_H_
#define FFT_UTILITY_H_

#include <fftw3.h>
#include <complex>

namespace oskar {

class FFTUtility
{
    public:
        static std::complex<float>* fftPhase(const unsigned nx, const unsigned ny,
                std::complex<float>* data);

        static fftwf_complex * fftPhase(const unsigned nx, const unsigned ny,
                fftwf_complex * data);

        static float * fftPhase(const unsigned nx, const unsigned ny,
                float * data);

        static float * fft_c2r_2d(const unsigned size, const std::complex<float> * in,
                float * out);
};



} // namespace oskar
#endif // FFT_UTILITY_H_
