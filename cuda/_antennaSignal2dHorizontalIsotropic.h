#ifndef OSKAR__ANTENNA_SIGNAL_2D_HORIZONTAL_ISOTROPIC_H
#define OSKAR__ANTENNA_SIGNAL_2D_HORIZONTAL_ISOTROPIC_H

#include "cuda/CudaEclipse.h"

/**
 * @file _antennaSignal2dHorizontalIsotropic.h
 */

/// CUDA kernel to compute antenna signals.
__global__
void _antennaSignal2dHorizontalIsotropic(const int na, const float* ax,
        const float* ay, const int ns, const float* samp, const float3* strig,
        const float k, float2* signals);

__global__
void _antennaSignal2dHorizontalIsotropicCached(const unsigned na,
        const float* ax, const float* ay, const unsigned ns, const float* samp,
        const float3* strig, const float k, const unsigned maxSourcesPerBlock,
        float2* signals);

#endif // OSKAR__ANTENNA_SIGNAL_2D_HORIZONTAL_ISOTROPIC_H
