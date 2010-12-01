#ifndef OSKAR__BEAM_PATTERN_2D_HORIZONTAL_GEOMETRIC_H
#define OSKAR__BEAM_PATTERN_2D_HORIZONTAL_GEOMETRIC_H

#include "cuda/CudaEclipse.h"

/**
 * @file _beamPattern2dHorizontalGeometric.h
 */

// Shared memory pointer used by the kernel.
extern __shared__ float2 sharedMem[];

/// CUDA kernel to compute a beam pattern directly, for the given direction.
__global__
void _beamPattern2dHorizontalGeometric(const int na, const float* ax,
        const float* ay, const float cosBeamEl, const float cosBeamAz,
        const float sinBeamAz, const int ns, const float* saz,
        const float* sel, const float k, float2* image);

#endif // OSKAR__BEAM_PATTERN_2D_HORIZONTAL_GEOMETRIC_H
