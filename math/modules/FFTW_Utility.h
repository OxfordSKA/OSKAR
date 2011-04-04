#ifndef FFTW_UTILITY_H_
#define FFTW_UTILITY_H_

#include <complex>
#include <fftw3.h>

typedef std::complex<float> Complex;

namespace oskar {

class FFT
{
    public:
        static Complex * fftPhase(const unsigned nx, const unsigned ny,
                Complex * data);

};



} // namespace oskar
#endif // FFTW_UTILITY_H_
