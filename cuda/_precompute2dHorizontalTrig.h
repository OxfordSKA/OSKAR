#ifndef OSKAR__PRECOMPUTE_2D_HORIZONTAL_TRIG_H
#define OSKAR__PRECOMPUTE_2D_HORIZONTAL_TRIG_H

#include "cuda/CudaEclipse.h"

/**
 * @file _precompute2dHorizontalTrig.h
 */

/// CUDA kernel to compute source position trigonometry.
__global__
void _precompute2dHorizontalTrig(const int ns, const float2* spos, float3* strig);

#endif // OSKAR__PRECOMPUTE_2D_HORIZONTAL_TRIG_H
