#ifndef OSKAR_CUDA_ANTENNA_SIGNALS_H
#define OSKAR_CUDA_ANTENNA_SIGNALS_H

/**
 * @file antennaSignals.h
 */

#include "cuda/CudaEclipse.h"
#include "cuda/CudaTypes.h"

/// Computes antenna signals using CUDA.
void antennaSignals(const int na, const float* ax, const float* ay,
        const int ns, const float* samp, const float* slon, const float* slat,
        const float k, float* signals);

#endif // OSKAR_CUDA_ANTENNA_SIGNALS_H
