#ifndef _ANTENNA_SIGNALS_H
#define _ANTENNA_SIGNALS_H

#include "cuda/CudaEclipse.h"

/**
 * @file _antennaSignals.h
 */

// Shared memory pointer used by the kernel.
extern __shared__ float2 sharedMem[];

/// CUDA kernel to compute antenna signals.
__global__
void _antennaSignals(const int na, const float* ax, const float* ay,
        const int ns, const float* samp, const float* slon, const float* slat,
        const float k, float2* signals);

#endif // _ANTENNA_SIGNALS_H
