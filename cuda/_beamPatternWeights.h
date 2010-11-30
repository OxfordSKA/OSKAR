#ifndef _BEAM_PATTERN_WEIGHTS_H
#define _BEAM_PATTERN_WEIGHTS_H

#include "cuda/CudaEclipse.h"

/**
 * @file _beamPatternWeights.h
 */

// Shared memory pointer used by the kernel.
extern __shared__ float2 sharedMem[];

/// CUDA kernel to compute a beam pattern using the given weights vector.
__global__
void _beamPatternWeights(const int na, const float* ax, const float* ay,
        const float2* weights, const int ns, const float* slon, const float* slat,
        const float k, float2* image);

#endif // _BEAM_PATTERN_WEIGHTS_H
