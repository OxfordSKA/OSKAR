#ifndef OSKAR__EQUATORIAL_TO_HORIZONTAL_GENERIC_H
#define OSKAR__EQUATORIAL_TO_HORIZONTAL_GENERIC_H

#include "cuda/CudaEclipse.h"

/**
 * @file _equatorialToHorizontalGeneric.h
 */

/// CUDA kernel to compute local source positions.
__global__
void _equatorialToHorizontalGeneric(const int ns, const float2* radec,
        const float cosLat, const float sinLat, const float lst, float2* azel);

#endif // OSKAR__EQUATORIAL_TO_HORIZONTAL_GENERIC_H
