#ifndef _BEAM_PATTERN_H
#define _BEAM_PATTERN_H

#include "cuda/CudaEclipse.h"

/**
 * @file _beamPattern.h
 */

/// CUDA kernel to compute a beam pattern.
__global__
void _beamPattern(const int na, const float* ax, const float* ay,
        const float2* weights, const int ns, const float* slon, const float* slat,
        const float k, float2* image);

#endif // _BEAM_PATTERN_H
