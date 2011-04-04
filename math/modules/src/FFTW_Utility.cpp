#include "math/modules/FFTW_Utility.h"

namespace oskar {

Complex * FFT::fftPhase(const unsigned nx, const unsigned ny, Complex * data)
{
    for (unsigned j = 0; j < ny; ++j)
    {
        for (unsigned i = 0; i < ny; ++i)
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


} // namespace oskar
