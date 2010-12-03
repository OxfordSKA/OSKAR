#ifndef OSKAR_CUDA_BEAMFORMER_MATRIX_VECTOR_H
#define OSKAR_CUDA_BEAMFORMER_MATRIX_VECTOR_H

/**
 * @file beamformerMatrixVector.h
 */

#include "cuda/CudaEclipse.h"
#include "cuda/CudaTypes.h"

/// Computes beams using CUDA.
void beamformerMatrixVector(const unsigned na, const unsigned nb,
        const float* signals, const float* weights, float* beams);

#endif // OSKAR_CUDA_BEAMFORMER_MATRIX_VECTOR_H
