/* Copyright (c) 2018-2020, The OSKAR Developers. See LICENSE file. */

#include "math/define_multiply.h"
#include "interferometer/define_jones_apply_station_gains.h"
#include "interferometer/define_evaluate_jones_K.h"
#include "interferometer/define_evaluate_jones_R.h"
#include "utility/oskar_cuda_registrar.h"
#include "utility/oskar_kernel_macros.h"
#include "utility/oskar_vector_types.h"

/* Kernels */

#define Real float
#define Real2 float2
#define Real4c float4c
#include "interferometer/src/oskar_interferometer_gpu.cl"
#include "interferometer/src/oskar_interferometer.cl"
#undef Real
#undef Real2
#undef Real4c

#define Real double
#define Real2 double2
#define Real4c double4c
#include "interferometer/src/oskar_interferometer_gpu.cl"
#include "interferometer/src/oskar_interferometer.cl"
#undef Real
#undef Real2
#undef Real4c
