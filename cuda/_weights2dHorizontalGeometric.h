#ifndef OSKAR__WEIGHTS_2D_HORIZONTAL_GEOMETRIC_H
#define OSKAR__WEIGHTS_2D_HORIZONTAL_GEOMETRIC_H

#include "cuda/CudaEclipse.h"

/**
 * @file _weights2dHorizontalGeometric.h
 */

/// CUDA kernel to generate beamforming weights.
__global__
void _weights2dHorizontalGeometric(const int na, const float* ax, const float* ay,
        const int nb, const float* cbe, const float* cba, const float* sba,
        const float k, float2* weights);

#endif // OSKAR__WEIGHTS_2D_HORIZONTAL_GEOMETRIC_H
