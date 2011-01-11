#ifndef OSKAR__BEAM_PATTERN_2D_HORIZONTAL_GEOMETRIC_H
#define OSKAR__BEAM_PATTERN_2D_HORIZONTAL_GEOMETRIC_H

#include "cuda/CudaEclipse.h"

/**
 * @file _beamPattern2dHorizontalGeometric.h
 */

/// CUDA kernel to compute a beam pattern directly, for the given direction.
__global__
void _beamPattern2dHorizontalGeometric(const unsigned na, const float* ax,
        const float* ay, const float cosBeamEl, const float cosBeamAz,
        const float sinBeamAz, const unsigned ns, const float* saz,
        const float* sel, const float k, const unsigned maxAntennasPerBlock,
        float2* image);

#endif // OSKAR__BEAM_PATTERN_2D_HORIZONTAL_GEOMETRIC_H
