/* Copyright (c) 2018-2020, The University of Oxford. See LICENSE file. */

#include "math/define_dft_c2r.h"
#include "math/define_dftw_c2c.h"
#include "math/define_dftw_m2m.h"
#include "math/define_fftphase.h"
#include "math/define_gaussian_circular.h"
#include "math/define_legendre_polynomial.h"
#include "math/define_multiply.h"
#include "math/define_prefix_sum.h"
#include "math/define_spherical_harmonic.h"
#include "utility/oskar_cuda_registrar.h"
#include "utility/oskar_kernel_macros.h"
#include "utility/oskar_vector_types.h"
#include <cuda_runtime.h>

/* Kernels */

#define Real float
#define Real2 float2
#define Real4c float4c
#include "math/src/oskar_math_gpu.cl"
#include "math/src/oskar_math.cl"
#undef Real
#undef Real2
#undef Real4c

#define Real double
#define Real2 double2
#define Real4c double4c
#include "math/src/oskar_math_gpu.cl"
#include "math/src/oskar_math.cl"
#undef Real
#undef Real2
#undef Real4c

#include "math/src/oskar_prefix_sum_gpu.cl"
