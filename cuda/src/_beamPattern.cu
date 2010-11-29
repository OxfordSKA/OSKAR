#include "cuda/_beamPattern.h"
#include "math/core/phase.h"

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
 * @param[in] ax Array of antenna x positions.
 * @param[in] ay Array of antenna y positions.
 * @param[in] weights Array of complex antenna weights (length na).
 * @param[in] ns The number of test source positions.
 * @param[in] slon The longitude coordinates of the test source.
 * @param[in] slat The latitude coordinates of the test source.
 * @param[in] k The wavenumber (rad / m).
 * @param[out] image The computed beam pattern (see note, above).
 */
__global__
void _beamPattern(const int na, const float* ax, const float* ay,
        const float2* weights, const int ns, const float* slon, const float* slat,
        const float k, float2* image)
{
    // Get the pixel (source position) ID that this thread is working on.
    const int s = blockDim.x * blockIdx.x + threadIdx.x;
    if (s >= ns) return; // Return if the index is out of range.

    // Get the source position.
    const float az = slon[s];
    const float el = slat[s];
    const float cosEl = cosf(el);
    const float sinAz = sinf(az);
    const float cosAz = cosf(az);

    // Loop over all antennas.
    image[s] = make_float2(0.0, 0.0);
    for (int a = 0; a < na; ++a) {
        // Calculate the geometric phase from the source.
        const float phase = GEOMETRIC_PHASE(ax[a], ay[a],
                cosEl, sinAz, cosAz, k);
        const float2 signal = make_float2(cosf(phase), sinf(phase));

        // Perform complex multiply-accumulate.
        const float2 w = weights[a];
        image[s].x += (signal.x * w.x - signal.y * w.y); // RE*RE - IM*IM
        image[s].y += (signal.y * w.x + signal.x * w.y); // IM*RE + RE*IM
    }
}

__global__
void _beamPattern2(const int na, const float* ax, const float* ay,
        const float2* weights, const int ns, const float* slon, const float* slat,
        const float k, float2* image)
{
    // Get the pixel (source position) ID that this thread is working on.
    const int s = blockDim.x * blockIdx.x + threadIdx.x;
    if (s >= ns) return; // Return if the index is out of range.

    // Get the source position.
    const float az = slon[s];
    const float el = slat[s];
    const float cosEl = cosf(el);
    const float sinAz = sinf(az);
    const float cosAz = cosf(az);

    // Loop over all antennas.
    __shared__ float2 temp[384];
    temp[threadIdx.x] = make_float2(0.0, 0.0);

    for (int a = 0; a < na; ++a) {
        // Calculate the geometric phase from the source.
        const float phase = GEOMETRIC_PHASE(ax[a], ay[a],
                cosEl, sinAz, cosAz, k);
        const float2 signal = make_float2(cosf(phase), sinf(phase));

        // Perform complex multiply-accumulate.
        const float2 w = weights[a];
        temp[threadIdx.x].x += (signal.x * w.x - signal.y * w.y); // RE*RE - IM*IM
        temp[threadIdx.x].y += (signal.y * w.x + signal.x * w.y); // IM*RE + RE*IM
    }

    image[s] = temp[threadIdx.x];
}


__global__
void _beamPattern3(const int na, const float* ax, const float* ay,
        const float cosBeamEl, const float cosBeamAz,  const float sinBeamAz,
        const int ns, const float* slon, const float* slat,
        const float k, float2* image)
{
    // Get the pixel (source position) ID that this thread is working on.
    const int s = blockDim.x * blockIdx.x + threadIdx.x;
    if (s >= ns) return; // Return if the index is out of range.

    // Get the source position.
    const float az = slon[s];
    const float el = slat[s];
    const float cosEl = cosf(el);
    const float sinAz = sinf(az);
    const float cosAz = cosf(az);

    // Loop over all antennas.
    sharedMem[threadIdx.x] = make_float2(0.0, 0.0);

    for (int a = 0; a < na; ++a) {
        // Calculate the geometric phase of the beam direction.
        const float phaseBeam = -GEOMETRIC_PHASE(ax[a], ay[a],
                cosBeamEl, sinBeamAz, cosBeamAz, k);
        float2 w;
        sincosf(phaseBeam, &w.x, &w.y);

        // Calculate the geometric phase from the source.
        const float phase = GEOMETRIC_PHASE(ax[a], ay[a],
                cosEl, sinAz, cosAz, k);
        float2 signal;
        sincosf(phase, &signal.x, &signal.y);

        // Perform complex multiply-accumulate.
        sharedMem[threadIdx.x].x += (signal.x * w.x - signal.y * w.y); // RE*RE - IM*IM
        sharedMem[threadIdx.x].y += (signal.y * w.x + signal.x * w.y); // IM*RE + RE*IM
    }

    image[s].x = sharedMem[threadIdx.x].x / na;
    image[s].y = sharedMem[threadIdx.x].y / na;
}

