#ifndef OSKAR__BEAM_PATTERN_2D_HORIZONTAL_WEIGHTS_H
#define OSKAR__BEAM_PATTERN_2D_HORIZONTAL_WEIGHTS_H

#include "cuda/CudaEclipse.h"

/**
 * @file _beamPattern2dHorizontalWeights.h
 */

/// CUDA kernel to compute a beam pattern using the given weights vector.
__global__
void _beamPattern2dHorizontalWeights(const int na, const float* ax,
        const float* ay, const float2* weights, const int ns,
        const float* saz, const float* sel, const float k,
        const unsigned maxAntennasPerBlock, float2* image);

#endif // OSKAR__BEAM_PATTERN_2D_HORIZONTAL_WEIGHTS_H
