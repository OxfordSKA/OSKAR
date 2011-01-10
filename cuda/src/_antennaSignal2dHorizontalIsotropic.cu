#include "cuda/_antennaSignal2dHorizontalIsotropic.h"
#include "math/core/phase.h"

/**
 * @details
 * This CUDA kernel evaluates the antenna signals for the given source and
 * antenna positions. It requires (8 * number_of_threads_per_block) bytes
 * of shared memory to be preallocated by the caller.
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

/**
 * @details
 * Same as above, but using manual caching of source data into shared memory.
 */
__global__
void _antennaSignal2dHorizontalIsotropicCached(const unsigned na,
        const float* ax, const float* ay, const unsigned ns, const float* samp,
        const float3* strig, const float k, const unsigned maxSourcesPerBlock,
        float2* signals)
{
    // Get the antenna ID that this thread is working on.
    const unsigned a = blockDim.x * blockIdx.x + threadIdx.x;
    if (a >= na) return;

    // Get the antenna position.
    float x = ax[a], y = ay[a];
    float phase, sinPhase, cosPhase;

    // Initialise shared memory to hold complex antenna signal.
    float2* lsignal = (float2*) sharedMem;
    float4* lsrc = (float4*) (&sharedMem[blockDim.x]);
    lsignal[threadIdx.x] = make_float2(0.0, 0.0);

    // Divide source list up into blocks, and cache the contents of each block
    // in shared memory before using it to accumulate the antenna signal.
    unsigned blocks = (ns + maxSourcesPerBlock - 1) / maxSourcesPerBlock;
    for (unsigned block = 0; block < blocks; ++block) {
        const unsigned sourceStart = block * maxSourcesPerBlock;
        unsigned sourcesInBlock = ns - sourceStart;
        if (sourcesInBlock > maxSourcesPerBlock) {
            sourcesInBlock = maxSourcesPerBlock;
        }

        // There are blockDim.x threads available - need to copy
        // sourcesInBlock pieces of data from global memory.
        for (unsigned t = threadIdx.x; t < sourcesInBlock; t += blockDim.x) {
            const unsigned sg = t + sourceStart; // global source index
            lsrc[t].x = strig[sg].x;
            lsrc[t].y = strig[sg].y;
            lsrc[t].z = strig[sg].z;
            lsrc[t].w = samp[sg];
        }

        // Must synchronise before computing the signal from these sources.
        __syncthreads();

        // Loop over sources in block.
        for (unsigned s = 0; s < sourcesInBlock; ++s) {
            // Calculate the geometric phase from the source.
            phase = GEOMETRIC_PHASE_2D_HORIZONTAL(x, y,
                    lsrc[s].z, lsrc[s].y, lsrc[s].x, k);

            // Perform complex multiply-accumulate.
            sincosf(phase, &sinPhase, &cosPhase);
            lsignal[threadIdx.x].x += (lsrc[s].w * cosPhase);
            lsignal[threadIdx.x].y += (lsrc[s].w * sinPhase);
        }

        // Must synchronise again before loading in a new block of sources.
        __syncthreads();
    }

    // Copy shared memory back into global memory.
    signals[a].x = lsignal[threadIdx.x].x;
    signals[a].y = lsignal[threadIdx.x].y;
}
