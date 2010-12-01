#include "cuda/_weights2dHorizontalGeometric.h"
#include "math/core/phase.h"

/**
 * @details
 * This CUDA kernel produces the complex antenna beamforming weights for the
 * given directions, and stores them in device memory.
 * Each thread generates the complex weights for a single antenna and a single
 * beam direction.
 *
 * The number of floating-point operations performed by this kernel is:
 * \li Sines and cosines: 2 * na * nb.
 * \li Multiplies: 4 * na * nb.
 * \li Divides: 2 * na * nb.
 * \li Additions / subtractions: na * nb.
 *
 * @param[in] na Number of antennas.
 * @param[in] ax Array of antenna x positions in metres.
 * @param[in] ay Array of antenna y positions in metres.
 * @param[in] nb Number of beams.
 * @param[in] cbe Cosine of all beam elevations.
 * @param[in] cba Cosine of all beam azimuths.
 * @param[in] sba Sine of all beam azimuths.
 * @param[in] k The wavenumber (rad / m).
 * @param[out] weights Matrix of complex antenna weights (na columns, nb rows).
 */
__global__
void _weights2dHorizontalGeometric(const int na, const float* ax, const float* ay,
        const int nb, const float* cbe, const float* cba, const float* sba,
        const float k, float2* weights)
{
    // Get the antenna and beam ID that this thread is working on.
    const int i = blockDim.x * blockIdx.x + threadIdx.x; // Thread index.
    const int a = i % na; // Antenna index.
    const int b = i / na; // Beam index.
    if (a >= na || b >= nb) return; // Return if either index is out of range.

    // Compute the geometric phase of the beam direction.
    const float phase = -GEOMETRIC_PHASE_2D_HORIZONTAL(ax[a], ay[a],
            cbe[b], sba[b], cba[b], k);
    const int w = a + b*na;
    weights[w].x = cosf(phase) / na; // Normalised real part.
    weights[w].y = sinf(phase) / na; // Normalised imaginary part.
}
