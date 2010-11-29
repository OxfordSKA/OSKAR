#include "cuda/beamPattern.h"
#include "cuda/_beamPattern.h"
#include "cuda/_generateWeights.h"

#define TIMER_ENABLE 1
#include "utility/timer.h"

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
//    int wThreadsPerBlock = 256;
//    int wBlocks = (na + wThreadsPerBlock - 1) / wThreadsPerBlock;
//    _generateWeights <<<wBlocks, wThreadsPerBlock>>> (
//            na, axd, ayd, weights, cosBeamEl, cosBeamAz, sinBeamAz, k);

    // Invoke kernel to compute the beam pattern on the device.
    int threadsPerBlock = 384;
    int blocks = (ns + threadsPerBlock - 1) / threadsPerBlock;
    TIMER_START
//    _beamPattern2 <<<blocks, threadsPerBlock>>> (na, axd, ayd, weights,
//            ns, slond, slatd, k, pix);
    _beamPattern3 <<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float2)>>> (na, axd, ayd, cosBeamEl, cosBeamAz, sinBeamAz,
            ns, slond, slatd, k, pix);
    cudaError_t err = cudaPeekAtLastError();
    const char* msg = cudaGetErrorString(err);
    printf("Error message: %s\n", msg);
    cudaThreadSynchronize();
    TIMER_STOP("Generated beam pattern")

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
