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
};



} // namespace oskar
#endif // FFT_UTILITY_H_
