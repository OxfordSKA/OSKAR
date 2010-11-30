#ifndef _BEAM_PATTERN_DIRECT_H
#define _BEAM_PATTERN_DIRECT_H

#include "cuda/CudaEclipse.h"

/**
 * @file _beamPatternDirect.h
 */

// Shared memory pointer used by the kernel.
extern __shared__ float2 sharedMem[];

/// CUDA kernel to compute a beam pattern directly, for the given direction.
__global__
void _beamPatternDirect(const int na, const float* ax, const float* ay,
        const float cosBeamEl, const float cosBeamAz,  const float sinBeamAz,
        const int ns, const float* slon, const float* slat,
        const float k, float2* image);

#endif // _BEAM_PATTERN_DIRECT_H
