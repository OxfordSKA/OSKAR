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
        const int maxAntennasPerBlock, float2* image)
{
    // Get the pixel (source position) ID that this thread is working on.
    const int s = blockDim.x * blockIdx.x + threadIdx.x;

    // Get the source position.
    // (NB. Cannot exit on index condition, as all threads are needed later).
    float az = 0.0f, el = 0.0f, sinAz, cosAz, cosEl;
    if (s < ns) {
        az = saz[s];
        el = sel[s];
    }
    cosEl = cosf(el);
    sincosf(az, &sinAz, &cosAz);

    // Initialise shared memory caches.
    // Antenna positions are cached as float2 for speed increase.
    float2* cpx = smem; // Cached pixel values.
    float2* cwt = cpx + blockDim.x; // Cached antenna weights.
    float2* cap = cwt + maxAntennasPerBlock; // Cached antenna positions.
    cpx[threadIdx.x] = make_float2(0.0f, 0.0f); // Clear pixel value.

    // Cache a block of antenna positions and weights into shared memory.
    int blocks = (na + maxAntennasPerBlock - 1) / maxAntennasPerBlock;
    for (int block = 0; block < blocks; ++block) {
        const int antennaStart = block * maxAntennasPerBlock;
        int antennasInBlock = na - antennaStart;
        if (antennasInBlock > maxAntennasPerBlock) {
            antennasInBlock = maxAntennasPerBlock;
        }

        // There are blockDim.x threads available - need to copy
        // antennasInBlock pieces of data from global memory.
        for (int t = threadIdx.x; t < antennasInBlock; t += blockDim.x) {
            const int ag = antennaStart + t; // Global antenna index.
            cwt[t] = weights[ag];
            cap[t].x = ax[ag];
            cap[t].y = ay[ag];
        }

        // Must synchronise before computing the signal for these antennas.
        __syncthreads();

        // Loop over antennas in block.
        for (int a = 0; a < antennasInBlock; ++a) {
            // Calculate the geometric phase from the source.
            float2 signal, w = cwt[a];
            float phaseSrc = GEOMETRIC_PHASE_2D_HORIZONTAL(cap[a].x,
                    cap[a].y, cosEl, sinAz, cosAz, k);
            __sincosf(phaseSrc, &signal.y, &signal.x);

            // Perform complex multiply-accumulate.
            cpx[threadIdx.x].x += (signal.x * w.x - signal.y * w.y);
            cpx[threadIdx.x].y += (signal.y * w.x + signal.x * w.y);
        }

        // Must synchronise again before loading in a new block of antennas.
        __syncthreads();
    }

    // Copy shared memory back into global memory.
    if (s < ns)
        image[s] = cpx[threadIdx.x];
}
