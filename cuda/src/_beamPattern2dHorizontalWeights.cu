#include "cuda/_beamPattern2dHorizontalWeights.h"
#include "math/core/phase.h"

// Shared memory pointer used by the kernel.
extern __shared__ float2 smem[];

/**
 * @details
 * This CUDA kernel evaluates the beam pattern for the given antenna
 * positions and weights vector, using the supplied positions of the test
 * source.
 *
 * Each thread evaluates a single pixel of the beam pattern, looping over
 * all the antennas while performing a complex multiply-accumulate with the
 * required beamforming weights.
 *
 * The computed beam pattern is returned in the \p image array, which
 * must be pre-sized to length 2*ns. The values in the \p image array
 * are alternate (real, imag) pairs for each test source position.
 *
 * The number of floating-point operations performed by this kernel is:
 * \li Sines and cosines: ns * (2 * na + 3).
 * \li Multiplies: 8 * ns * na.
 * \li Additions / subtractions: 5 * ns * na.
 *
 * @param[in] na Number of antennas.
 * @param[in] ax Array of antenna x positions in metres.
 * @param[in] ay Array of antenna y positions in metres.
 * @param[in] weights Array of complex antenna weights (length na).
 * @param[in] ns The number of test source positions.
 * @param[in] saz The azimuth coordinates of the test source in radians.
 * @param[in] sel The elevation coordinates of the test source in radians.
 * @param[in] k The wavenumber (rad / m).
 * @param[out] image The computed beam pattern (see note, above).
 */
__global__
void _beamPattern2dHorizontalWeights(const int na, const float* ax,
        const float* ay, const float2* weights, const int ns,
        const float* saz, const float* sel, const float k,
        const unsigned maxAntennasPerBlock, float2* image)
{
    // Get the pixel (source position) ID that this thread is working on.
    const int s = blockDim.x * blockIdx.x + threadIdx.x;
    if (s >= ns) return; // Return if the index is out of range.

    // Get the source position.
    const float az = saz[s];
    const float el = sel[s];
    const float cosEl = cosf(el);
    const float sinAz = sinf(az);
    const float cosAz = cosf(az);

    // Initialise shared memory cache to hold complex pixel amplitude.
    float2* lpixel = smem;
    float2* lant = (float2*) (&smem[blockDim.x]);
    float2* lweight = (float2*) (&smem[blockDim.x + maxAntennasPerBlock]);
    lpixel[threadIdx.x] = make_float2(0.0, 0.0);
    float2 w, signal;
    float phaseSrc = 0.0;

    // Cache a block of antenna positions and weights into shared memory.
    unsigned blocks = (na + maxAntennasPerBlock - 1) / maxAntennasPerBlock;
    for (unsigned block = 0; block < blocks; ++block) {
        const unsigned antennaStart = block * maxAntennasPerBlock;
        unsigned antennasInBlock = na - antennaStart;
        if (antennasInBlock > maxAntennasPerBlock) {
            antennasInBlock = maxAntennasPerBlock;
        }

        // There are blockDim.x threads available - need to copy
        // antennasInBlock pieces of data from global memory.
        for (unsigned t = threadIdx.x; t < antennasInBlock; t += blockDim.x) {
            const unsigned ag = antennaStart + t; // Global antenna index.
            lant[t].x = ax[ag];
            lant[t].y = ay[ag];
            lweight[t] = weights[ag];
        }

        // Must synchronise before computing the signal for these antennas.
        __syncthreads();

        // Loop over antennas in block.
        for (unsigned a = 0; a < antennasInBlock; ++a) {
            // Calculate the geometric phase from the source.
            phaseSrc = GEOMETRIC_PHASE_2D_HORIZONTAL(lant[a].x, lant[a].y,
                    cosEl, sinAz, cosAz, k);
            sincosf(phaseSrc, &signal.y, &signal.x);

            // Perform complex multiply-accumulate.
            w = lweight[a];
            lpixel[threadIdx.x].x += (signal.x * w.x - signal.y * w.y);
            lpixel[threadIdx.x].y += (signal.y * w.x + signal.x * w.y);
        }

        // Must synchronise again before loading in a new block of antennas.
        __syncthreads();
    }

    // Copy shared memory back into global memory.
    image[s].x = lpixel[threadIdx.x].x;
    image[s].y = lpixel[threadIdx.x].y;
}
