#ifndef _GENERATE_WEIGHTS_H
#define _GENERATE_WEIGHTS_H

#include "cuda/CudaEclipse.h"

/**
 * @file _generateWeights.h
 */

/// CUDA kernel to generate beamforming weights.
__global__
void _generateWeights(const int na, const float* ax, const float* ay,
        const int nb, const float* cbe, const float* cba, const float* sba,
        const float k, float2* weights);

#endif // _GENERATE_WEIGHTS_H
