#include "cuda/beamPattern.h"

#ifndef M_PI
#define M_PI 3.1415926535
#endif

#define DEG2RAD (M_PI / 180.0)
#define RAD2DEG (180.0 / M_PI)
#define C_0 299792458.0


// Macro used to compute the geometric phase.
#define GEOMETRIC_PHASE(x, y, cosEl, sinAz, cosAz, k) \
    (-k * cosEl * (x * sinAz + y * cosAz))


// Kernel function prototypes.
__global__ void _generateWeights(const int na, const float* ax, const float* ay,
        float2* weights, const float cosBeamEl, const float cosBeamAz,
        const float sinBeamAz, const float k);
__global__ void _beamPattern(const int na, const float* ax, const float* ay,
        const float2* weights, const int ns, const float* slon, const float* slat,
        const float k, float2* image);


/**
 * @details
 * Computes a beam pattern using CUDA.
 *
 * The function must be supplied with the antenna x- and y-positions, the
 * test source longitude and latitude positions, the beam direction, and
 * the wavenumber.
 *
 * The computed beam pattern is returned in the \p image array, which
 * must be pre-sized to length 2*ns. The values in the \p image array
 * are alternate (real, imag) pairs for each position of the test source.
 *
 * @param[in] na The number of antennas.
 * @param[in] ax The antenna x-positions in metres.
 * @param[in] ay The antenna y-positions in metres.
 * @param[in] ns The number of test source positions.
 * @param[in] slon The longitude coordinates of the test source.
 * @param[in] slat The latitude coordinates of the test source.
 * @param[in] ba The beam azimuth direction in radians
 * @param[in] be The beam elevation direction in radians.
 * @param[in] k The wavenumber (rad / m).
 * @param[out] image The computed beam pattern (see note, above).
 */
void beamPattern(const int na, const float* ax, const float* ay,
        const int ns, const float* slon, const float* slat,
        const float ba, const float be, const float k,
        float* image)
{
    // Precompute.
    float sinBeamAz = sin(ba);
    float cosBeamAz = cos(ba);
    float cosBeamEl = cos(be);

    // Allocate memory for antenna positions, antenna weights,
    // test source positions and pixel values on the device.
    float *axd, *ayd, *slond, *slatd;
    float2 *weights, *pix;
    cudaMalloc((void**)&axd, na * sizeof(float));
    cudaMalloc((void**)&ayd, na * sizeof(float));
    cudaMalloc((void**)&weights, na * sizeof(float2));
    cudaMalloc((void**)&slond, ns * sizeof(float));
    cudaMalloc((void**)&slatd, ns * sizeof(float));
    cudaMalloc((void**)&pix, ns * sizeof(float2));

    // Copy antenna positions and test source positions to device.
    cudaMemcpy(axd, ax, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ayd, ay, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(slond, slon, ns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(slatd, slat, ns * sizeof(float), cudaMemcpyHostToDevice);

    // Invoke kernel to compute antenna weights on the device.
    int wThreadsPerBlock = 256;
    int wBlocks = (na + wThreadsPerBlock - 1) / wThreadsPerBlock;
    _generateWeights <<<wBlocks, wThreadsPerBlock>>> (
            na, axd, ayd, weights, cosBeamEl, cosBeamAz, sinBeamAz, k);

    // Invoke kernel to compute the beam pattern on the device.
    int threadsPerBlock = 384;
    int blocks = (ns + threadsPerBlock - 1) / threadsPerBlock;
    _beamPattern <<<blocks, threadsPerBlock>>> (na, axd, ayd, weights,
            ns, slond, slatd, k, pix);

    // Copy result from device memory to host memory.
    cudaMemcpy(image, pix, ns * sizeof(float2), cudaMemcpyDeviceToHost);

    // Free device memory.
    cudaFree(axd);
    cudaFree(ayd);
    cudaFree(weights);
    cudaFree(slond);
    cudaFree(slatd);
    cudaFree(pix);
}


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
__global__ void _generateWeights(const int na, const float* ax, const float* ay,
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
__global__ void _beamPattern(const int na, const float* ax, const float* ay,
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

