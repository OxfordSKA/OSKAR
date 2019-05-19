/* Copyright (c) 2018, The University of Oxford. See LICENSE file. */

#include "splines/define_dierckx_bispev_bicubic.h"
#include "utility/oskar_cuda_registrar.h"
#include "utility/oskar_kernel_macros.h"
#include "utility/oskar_vector_types.h"
#include <cuda_runtime.h>

/* Kernels */

#define Real float
#include "splines/src/oskar_splines.cl"
#undef Real

#define Real double
#include "splines/src/oskar_splines.cl"
#undef Real
