#ifndef _GENERATE_WEIGHTS_H
#define _GENERATE_WEIGHTS_H

#include "cuda/CudaEclipse.h"

/**
 * @file _generateWeights.h
 */

/// CUDA kernel to generate beamforming weights.
__global__
void _generateWeights(const int na, const float* ax, const float* ay,
        float2* weights, const float cosBeamEl, const float cosBeamAz,
        const float sinBeamAz, const float k);

#endif // _GENERATE_WEIGHTS_H
