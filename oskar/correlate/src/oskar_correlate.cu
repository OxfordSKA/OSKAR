/* Copyright (c) 2018-2019, The University of Oxford. See LICENSE file. */

#include "correlate/define_auto_correlate.h"
#include "correlate/define_correlate_utils.h"
#include "correlate/define_evaluate_auto_power.h"
#include "correlate/define_evaluate_cross_power.h"
#include "math/define_multiply.h"
#include "utility/oskar_cuda_registrar.h"
#include "utility/oskar_kernel_macros.h"
#include "utility/oskar_vector_types.h"

/* Kernels */

#define Real float
#define Real2 float2
#define Real4c float4c
#include "correlate/src/oskar_correlate_gpu.cl"
#include "correlate/src/oskar_correlate.cl"
#undef Real
#undef Real2
#undef Real4c

#define Real double
#define Real2 double2
#define Real4c double4c
#include "correlate/src/oskar_correlate_gpu.cl"
#include "correlate/src/oskar_correlate.cl"
#undef Real
#undef Real2
#undef Real4c
