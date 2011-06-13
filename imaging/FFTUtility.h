#ifndef FFT_UTILITY_H_
#define FFT_UTILITY_H_

#include "imaging/oskar_types.h"
#include <fftw3.h>

namespace oskar {

class FFTUtility
{
    public:
        static Complex * fftPhase(const unsigned nx, const unsigned ny,
                Complex * data);

        static fftwf_complex * fftPhase(const unsigned nx, const unsigned ny,
                fftwf_complex * data);

        static float * fftPhase(const unsigned nx, const unsigned ny,
                float * data);

        static float * fft_c2r_2d(const unsigned size, const Complex * in,
                float * out);
};



} // namespace oskar
#endif // FFT_UTILITY_H_
