#include "cuda/_beamPattern2dHorizontalGeometric.h"
#include "math/core/phase.h"

// Shared memory pointer used by the kernel.
extern __shared__ float2 smem[];

/**
 * @details
 * This CUDA kernel evaluates the beam pattern for the given antenna
 * positions and beam direction, using the supplied positions of the test
 * source.
 *
 * Each thread evaluates a single pixel of the beam pattern, looping over
 * all the antennas while performing a complex multiply-accumulate with the
 * required beamforming weights, which are computed by the kernel for the
 * given beam direction.
 *
 * The computed beam pattern is returned in the \p image array, which
 * must be pre-sized to length 2*ns. The values in the \p image array
 * are alternate (real, imag) pairs for each test source position.
 *
 * @param[in] na Number of antennas.
 * @param[in] ax Array of antenna x positions in metres.
 * @param[in] ay Array of antenna y positions in metres.
 * @param[in] cosBeamEl Cosine of beam elevation.
 * @param[in] cosBeamAz Cosine of beam azimuth.
 * @param[in] sinBeamAz Sine of beam azimuth.
 * @param[in] ns The number of test source positions.
 * @param[in] saz The azimuth coordinates of the test source in radians.
 * @param[in] sel The elevation coordinates of the test source in radians.
 * @param[in] k The wavenumber (rad / m).
 * @param[out] image The computed beam pattern (see note, above).
 */
__global__
void _beamPattern2dHorizontalGeometric(const int na, const float* ax,
        const float* ay, const float cosBeamEl, const float cosBeamAz,
        const float sinBeamAz, const int ns, const float* saz,
        const float* sel, const float k, const int maxAntennasPerBlock,
        float2* image)
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

    // Initialise shared memory cache to hold complex pixel amplitude.
    float2* cpx = smem; // Cached pixel values.
    float2* cap = (float2*) (&smem[blockDim.x]); // Cached antenna positions.
    cpx[threadIdx.x] = make_float2(0.0f, 0.0f);  // Clear pixel value.
    float2 w, signal;

    // Cache a block of antenna positions into shared memory.
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
            cap[t].x = ax[antennaStart + t];
            cap[t].y = ay[antennaStart + t];
        }

        // Must synchronise before computing the signal for these antennas.
        __syncthreads();

        // Loop over antennas in block.
        for (int a = 0; a < antennasInBlock; ++a) {
            // Get the antenna position from shared memory.
            const float x = cap[a].x;
            const float y = cap[a].y;

            // Calculate the geometric phase of the beam direction.
            const float phaseBeam = -GEOMETRIC_PHASE_2D_HORIZONTAL(x, y,
                    cosBeamEl, sinBeamAz, cosBeamAz, k);
            __sincosf(phaseBeam, &w.y, &w.x);

            // Calculate the geometric phase from the source.
            const float phaseSrc = GEOMETRIC_PHASE_2D_HORIZONTAL(x, y,
                    cosEl, sinAz, cosAz, k);
            __sincosf(phaseSrc, &signal.y, &signal.x);

            // Perform complex multiply-accumulate.
            cpx[threadIdx.x].x += (signal.x * w.x - signal.y * w.y);
            cpx[threadIdx.x].y += (signal.y * w.x + signal.x * w.y);
        }

        // Must synchronise again before loading in a new block of antennas.
        __syncthreads();
    }

    // Copy shared memory back into global memory.
    if (s < ns) {
        image[s].x = cpx[threadIdx.x].x / na;
        image[s].y = cpx[threadIdx.x].y / na;
    }
}
