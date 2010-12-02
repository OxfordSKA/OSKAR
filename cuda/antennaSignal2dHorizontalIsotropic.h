#ifndef OSKAR_CUDA_ANTENNA_SIGNAL_2D_HORIZONTAL_ISOTROPIC_H
#define OSKAR_CUDA_ANTENNA_SIGNAL_2D_HORIZONTAL_ISOTROPIC_H

/**
 * @file antennaSignal2dHorizontalIsotropic.h
 */

#include "cuda/CudaEclipse.h"
#include "cuda/CudaTypes.h"

/// Computes antenna signals using CUDA.
void antennaSignal2dHorizontalIsotropic(const unsigned na, const float* ax,
        const float* ay, const unsigned ns, const float* samp,
        const float* slon, const float* slat, const float k, float* signals);

#endif // OSKAR_CUDA_ANTENNA_SIGNAL_2D_HORIZONTAL_ISOTROPIC_H
