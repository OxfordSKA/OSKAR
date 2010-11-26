#ifndef OSKAR_CUDA_BEAM_PATTERN_H
#define OSKAR_CUDA_BEAM_PATTERN_H

/**
 * @file beamPattern.h
 */

#include "cuda/CudaEclipse.h"
#include "cuda/CudaTypes.h"

/// Computes a beam pattern using CUDA.
void beamPattern(const int na, const float* ax, const float* ay,
        const int ns, const float* slon, const float* slat,
        const float ba, const float be, const float k,
        float* image);

#endif // OSKAR_CUDA_BEAM_PATTERN_H
