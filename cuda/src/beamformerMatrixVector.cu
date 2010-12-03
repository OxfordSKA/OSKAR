#include "cuda/beamformerMatrixVector.h"
#include <cstdio>
#include <cublas.h>

/**
 * @details
 * Computes beams using CUDA.
 *
 * The computed beams are returned in the \p beams array, which
 * must be pre-sized to length 2*nb. The values in the \p beams array
 * are alternate (real, imag) pairs for each beam.
 *
 * @param[in] na The number of antennas.
 * @param[in] nb The number of beams to form.
 * @param[in] signals The vector of complex antenna signals (length na).
 * @param[in] weights The matrix of complex beamforming weights
 *                    (na columns, nb rows).
 * @param[out] beams The complex vector of output beams (length nb).
 */
void beamformerMatrixVector(const unsigned na, const unsigned nb,
        const float* signals, const float* weights, float* beams)
{
    // Initialise cuBLAS.
    cublasInit();

    // Allocate memory for antenna signals and beamforming weights
    // on the device.
    float2 *signalsd, *weightsd, *beamsd;
    cudaMalloc((void**)&signalsd, na * sizeof(float2));
    cudaMalloc((void**)&beamsd, nb * sizeof(float2));
    cudaMalloc((void**)&weightsd, na * nb * sizeof(float2));

    // Copy antenna signals and beamforming weights to the device.
    cudaMemcpy(signalsd, signals, na * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(weightsd, weights, na * nb * sizeof(float2), cudaMemcpyHostToDevice);

    // Call cuBLAS function to perform the matrix-vector multiplication.
    // Note that cuBLAS calls use Fortran-ordering (column major) for their
    // matrices, so we use the transpose here.
    cublasCgemv('t', na, nb, make_float2(1.0, 0.0),
            weightsd, na, signalsd, 1, make_float2(0.0, 0.0), beamsd, 1);

    // Copy result from device memory to host memory.
    cudaMemcpy(beams, beamsd, nb * sizeof(float2), cudaMemcpyDeviceToHost);

    // Free device memory.
    cudaFree(signalsd);
    cudaFree(weightsd);
    cudaFree(beamsd);

    // Shut down cuBLAS.
    cublasShutdown();
}
