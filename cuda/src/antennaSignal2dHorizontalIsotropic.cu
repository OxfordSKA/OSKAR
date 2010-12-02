#include "cuda/antennaSignal2dHorizontalIsotropic.h"
#include "cuda/_antennaSignal2dHorizontalIsotropic.h"
#include "cuda/_precompute2dHorizontalTrig.h"
#include <vector>
#include <cstdio>

/**
 * @details
 * Computes antenna signals using CUDA.
 *
 * The function must be supplied with the antenna x- and y-positions, the
 * source amplitudes, longitude and latitude positions, and the wavenumber.
 *
 * The computed antenna signals are returned in the \p signals array, which
 * must be pre-sized to length 2*na. The values in the \p signals array
 * are alternate (real, imag) pairs for each antenna.
 *
 * @param[in] na The number of antennas.
 * @param[in] ax The antenna x-positions in metres.
 * @param[in] ay The antenna y-positions in metres.
 * @param[in] ns The number of source positions.
 * @param[in] samp The source amplitudes.
 * @param[in] slon The source longitude coordinates in radians.
 * @param[in] slat The source latitude coordinates in radians.
 * @param[in] k The wavenumber (rad / m).
 * @param[out] signals The computed antenna signals (see note, above).
 */
void antennaSignal2dHorizontalIsotropic(const unsigned na, const float* ax,
        const float* ay, const unsigned ns, const float* samp,
        const float* slon, const float* slat, const float k, float* signals)
{
    // Create source position pairs in host memory.
    std::vector<float2> spos(ns);
    for (unsigned i = 0; i < ns; ++i) spos[i] = make_float2(slon[i], slat[i]);

    // Allocate memory for antenna positions, source positions
    // and antenna signals on the device.
    float *axd, *ayd, *sampd;
    float2 *sig, *sposd;
    float3 *strigd;
    cudaMalloc((void**)&axd, na * sizeof(float));
    cudaMalloc((void**)&ayd, na * sizeof(float));
    cudaMalloc((void**)&sampd, ns * sizeof(float));
    cudaMalloc((void**)&sig, na * sizeof(float2));
    cudaMalloc((void**)&sposd, ns * sizeof(float2));
    cudaMalloc((void**)&strigd, ns * sizeof(float3));

    // Copy antenna positions and test source positions to device.
    cudaMemcpy(axd, ax, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ayd, ay, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(sposd, &spos[0], ns * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(sampd, samp, ns * sizeof(float), cudaMemcpyHostToDevice);

    // Error code.
    cudaError_t err;

    // Invoke kernel to precompute source positions on the device.
    unsigned sThreadsPerBlock = 384;
    unsigned sBlocks = (ns + sThreadsPerBlock - 1) / sThreadsPerBlock;
    _precompute2dHorizontalTrig <<<sBlocks, sThreadsPerBlock>>>
            (ns, sposd, strigd);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess)
        printf("CUDA Error: %s\n", cudaGetErrorString(err));

    // Invoke kernel to compute antenna signals on the device.
    unsigned threadsPerBlock = 384;
    unsigned blocks = (na + threadsPerBlock - 1) / threadsPerBlock;
//    size_t sharedMem = threadsPerBlock * sizeof(float2);
    unsigned maxSourcesPerBlock = 384;
    size_t sharedMem = threadsPerBlock * sizeof(float2)
            + maxSourcesPerBlock * sizeof(float4);
    _antennaSignal2dHorizontalIsotropicCached <<<blocks, threadsPerBlock, sharedMem>>>
            (na, axd, ayd, ns, sampd, strigd, k, maxSourcesPerBlock, sig);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess)
        printf("CUDA Error: %s\n", cudaGetErrorString(err));

    // Copy result from device memory to host memory.
    cudaMemcpy(signals, sig, na * sizeof(float2), cudaMemcpyDeviceToHost);

    // Free device memory.
    cudaFree(axd);
    cudaFree(ayd);
    cudaFree(sampd);
    cudaFree(sig);
    cudaFree(sposd);
    cudaFree(strigd);
}
