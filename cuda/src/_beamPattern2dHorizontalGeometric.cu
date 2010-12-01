#include "cuda/_beamPattern2dHorizontalGeometric.h"
#include "math/core/phase.h"

/**
 * @details
 * This CUDA kernel evaluates the beam pattern for the given antenna
 * positions and beam direction, using the supplied positions of the test
 * source.
 *
 * Each thread evaluates a single pixel of the beam pattern, looping over
 * all the antennas while performing a complex multiply-accumulate with the
 * required beamforming weights, which are computed by the kernel for the
 * given beam direction. In some cases (using older cards), this is faster
 * than looking up the weights in global memory.
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
        const float* sel, const float k, float2* image)
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

    // Initialise shared memory to hold complex pixel amplitude.
    sharedMem[threadIdx.x] = make_float2(0.0, 0.0);
    float2 w, signal;
    float phaseBeam, phaseSrc, x, y;

    // Loop over all antennas.
    for (int a = 0; a < na; ++a) {
        // Get the antenna position from global memory.
        x = ax[a]; y = ay[a];

        // Calculate the geometric phase of the beam direction.
        // (Faster to recompute it here than look it up from global memory.)
        phaseBeam = -GEOMETRIC_PHASE_2D_HORIZONTAL(x, y,
                cosBeamEl, sinBeamAz, cosBeamAz, k);
        sincosf(phaseBeam, &w.y, &w.x);

        // Calculate the geometric phase from the source.
        phaseSrc = GEOMETRIC_PHASE_2D_HORIZONTAL(x, y, cosEl, sinAz, cosAz, k);
        sincosf(phaseSrc, &signal.y, &signal.x);

        // Perform complex multiply-accumulate.
        sharedMem[threadIdx.x].x += (signal.x * w.x - signal.y * w.y); // RE*RE - IM*IM
        sharedMem[threadIdx.x].y += (signal.y * w.x + signal.x * w.y); // IM*RE + RE*IM
    }

    // Copy shared memory back into global memory.
    image[s].x = sharedMem[threadIdx.x].x / na;
    image[s].y = sharedMem[threadIdx.x].y / na;
}
