#include "cuda/antennaSignal2dHorizontalIsotropic.h"
#include "cuda/_antennaSignal2dHorizontalIsotropic.h"

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
void antennaSignal2dHorizontalIsotropic(const int na, const float* ax,
        const float* ay, const int ns, const float* samp, const float* slon,
        const float* slat, const float k, float* signals)
{
    // Allocate memory for antenna positions, source positions
    // and antenna signals on the device.
    float *axd, *ayd, *slond, *slatd, *sampd;
    float2 *sig;
    cudaMalloc((void**)&axd, na * sizeof(float));
    cudaMalloc((void**)&ayd, na * sizeof(float));
    cudaMalloc((void**)&sampd, ns * sizeof(float));
    cudaMalloc((void**)&slond, ns * sizeof(float));
    cudaMalloc((void**)&slatd, ns * sizeof(float));
    cudaMalloc((void**)&sig, ns * sizeof(float2));

    // Copy antenna positions and test source positions to device.
    cudaMemcpy(axd, ax, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ayd, ay, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(slond, slon, ns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(slatd, slat, ns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(sampd, samp, ns * sizeof(float), cudaMemcpyHostToDevice);

    // Invoke kernel to compute antenna signals on the device.
    int threadsPerBlock = 384;
    int blocks = (ns + threadsPerBlock - 1) / threadsPerBlock;
//    _antennaSignal2dHorizontalIsotropic <<<blocks, threadsPerBlock>>> (na, axd, ayd,
//            ns, sampd, slond, slatd, k, sig);

    // Copy result from device memory to host memory.
    cudaMemcpy(signals, sig, ns * sizeof(float2), cudaMemcpyDeviceToHost);

    // Free device memory.
    cudaFree(axd);
    cudaFree(ayd);
    cudaFree(sampd);
    cudaFree(slond);
    cudaFree(slatd);
    cudaFree(sig);
}
