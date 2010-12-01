#include "cuda/_antennaSignal2dHorizontalIsotropic.h"
#include "math/core/phase.h"

/**
 * @details
 * This CUDA kernel evaluates the antenna signals for the given source and
 * antenna positions.
 *
 * Each thread evaluates the signal for a single antenna, looping over
 * all the sources.
 *
 * The cosine and sine of the source azimuths, and the cosine
 * of the elevations, must be given as triplets in the \p strig array:
 *
 * strig.x = {cosine azimuth}
 * strig.y = {sine azimuth}
 * strig.z = {cosine elevation}
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
 * @param[in] ax Array of antenna x positions in metres.
 * @param[in] ay Array of antenna y positions in metres.
 * @param[in] ns The number of source positions.
 * @param[in] samp The source amplitudes.
 * @param[in] strig The cosine and sine of the source coordinates.
 * @param[in] k The wavenumber (rad / m).
 * @param[out] signals The computed antenna signals (see note, above).
 */
__global__
void _antennaSignal2dHorizontalIsotropic(const int na, const float* ax,
        const float* ay, const int ns, const float* samp, const float3* strig,
        const float k, float2* signals)
{
    // Get the antenna ID that this thread is working on.
    const int a = blockDim.x * blockIdx.x + threadIdx.x;
    if (a >= na) return; // Return if the index is out of range.

    // Get the antenna position.
    const float x = ax[a];
    const float y = ay[a];

    // Initialise shared memory to hold complex antenna signal.
    sharedMem[threadIdx.x] = make_float2(0.0, 0.0);

    // Loop over all sources.
    float sinPhase, cosPhase;
    for (int s = 0; s < ns; ++s) {
        // Calculate the geometric phase from the source.
        const float phase = GEOMETRIC_PHASE_2D_HORIZONTAL(x, y,
                strig[s].z, strig[s].y, strig[s].x, k);

        // Perform complex multiply-accumulate.
        sincosf(phase, &sinPhase, &cosPhase);
        sharedMem[threadIdx.x].x += (samp[s] * cosPhase);
        sharedMem[threadIdx.x].y += (samp[s] * sinPhase);
    }

    // Copy shared memory back into global memory.
    signals[a].x = sharedMem[threadIdx.x].x;
    signals[a].y = sharedMem[threadIdx.x].y;
}

