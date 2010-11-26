#include "cuda/_antennaSignals.h"
#include "math/core/phase.h"

/**
 * @details
 * This CUDA kernel evaluates the antenna signals for the given source and
 * antenna positions.
 *
 * Each thread evaluates the signal for a single antenna, looping over
 * all the sources.
 *
 * The computed antenna signals are returned in the \p signals array, which
 * must be pre-sized to length 2*na. The values in the \p signals array
 * are alternate (real, imag) pairs for each antenna.
 *
 * The number of floating-point operations performed by this kernel is:
 * \li Sines and cosines:
 * \li Multiplies:
 * \li Additions / subtractions:
 *
 * @param[in] na Number of antennas.
 * @param[in] ax Array of antenna x positions.
 * @param[in] ay Array of antenna y positions.
 * @param[in] ns The number of source positions.
 * @param[in] samp The source amplitudes.
 * @param[in] slon The source longitude coordinates in radians.
 * @param[in] slat The source latitude coordinates in radians.
 * @param[in] k The wavenumber (rad / m).
 * @param[out] signals The computed antenna signals (see note, above).
 */
__global__
void _antennaSignals(const int na, const float* ax, const float* ay,
        const int ns, const float* samp, const float* slon, const float* slat,
        const float k, float2* signals)
{
    // Get the antenna ID that this thread is working on.
    const int a = blockDim.x * blockIdx.x + threadIdx.x;
    if (a >= na) return; // Return if the index is out of range.

    // Loop over all sources.
    signals[a] = make_float2(0.0, 0.0);
    for (int s = 0; s < ns; ++s) {
        // Get the source position (could do with major optimisation here!).
        const float az = slon[s];
        const float el = slat[s];
        const float cosAz = cosf(az);
        const float sinAz = sinf(az);
        const float cosEl = cosf(el);

        // Calculate the geometric phase from the source.
        const float phase = GEOMETRIC_PHASE(ax[a], ay[a],
                cosEl, sinAz, cosAz, k);

        // Perform complex multiply-accumulate.
        signals[a].x += (samp[s] * cosf(phase));
        signals[a].y += (samp[s] * sinf(phase));
    }
}

