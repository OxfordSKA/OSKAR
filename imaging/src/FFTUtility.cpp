#include "imaging/FFTUtility.h"

namespace oskar {

Complex * FFTUtility::fftPhase(const unsigned nx, const unsigned ny,
        Complex * data)
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
