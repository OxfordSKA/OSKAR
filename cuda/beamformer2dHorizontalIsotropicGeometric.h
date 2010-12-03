#ifndef OSKAR_CUDA_BEAMFORMER_2D_HORIZONTAL_ISOTROPIC_GEOMETRIC_H
#define OSKAR_CUDA_BEAMFORMER_2D_HORIZONTAL_ISOTROPIC_GEOMETRIC_H

/**
 * @file beamformer2dHorizontalIsotropicGeometric.h
 */

#include "cuda/CudaEclipse.h"
#include "cuda/CudaTypes.h"

/// Computes beams using CUDA.
void beamformer2dHorizontalIsotropicGeometric(const unsigned na,
        const float* ax, const float* ay, const unsigned ns, const float* samp,
        const float* slon, const float* slat, const unsigned nb,
        const float* blon, const float* blat, const float k, float* beams);

#endif // OSKAR_CUDA_BEAMFORMER_2D_HORIZONTAL_ISOTROPIC_GEOMETRIC_H
