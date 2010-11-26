#include "cuda/_generateWeights.h"
#include "math/core/phase.h"

/**
 * @details
 * This CUDA kernel produces the complex antenna beamforming weights for the
 * given direction, and stores them in device memory.
 * Each thread generates the complex weight for a single antenna.
 *
 * The number of floating-point operations performed by this kernel is:
 * \li Sines and cosines: 2 * na.
 * \li Multiplies: 4 * na.
 * \li Divides: 2 * na.
 * \li Additions / subtractions: na.
 *
 * @param[in] na Number of antennas.
 * @param[in] ax Array of antenna x positions.
 * @param[in] ay Array of antenna y positions.
 * @param[out] weights Array of generated complex antenna weights (length na).
 * @param[in] cosBeamEl Cosine of the beam elevation.
 * @param[in] cosBeamAz Cosine of the beam azimuth.
 * @param[in] sinBeamAz Sine of the beam azimuth.
 * @param[in] k Wavenumber.
 */
__global__
void _generateWeights(const int na, const float* ax, const float* ay,
        float2* weights, const float cosBeamEl, const float cosBeamAz,
        const float sinBeamAz, const float k)
{
    // Get the antenna ID that this thread is working on.
    const int a = blockDim.x * blockIdx.x + threadIdx.x;
    if (a >= na) return; // Return if the index is out of range.

    // Compute the geometric phase of the beam direction.
    const float phase = -GEOMETRIC_PHASE(ax[a], ay[a],
            cosBeamEl, sinBeamAz, cosBeamAz, k);
    weights[a].x = cosf(phase) / na; // Normalised real part.
    weights[a].y = sinf(phase) / na; // Normalised imaginary part.
}
