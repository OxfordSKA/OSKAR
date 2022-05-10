/* Copyright (c) 2018-2022, The OSKAR Developers. See LICENSE file. */

#include "math/define_multiply.h"
#include "mem/define_mem_add.h"
#include "mem/define_mem_conjugate.h"
#include "mem/define_mem_multiply.h"
#include "mem/define_mem_normalise.h"
#include "mem/define_mem_scale_real.h"
#include "mem/define_mem_set_value_real.h"
#include "utility/oskar_cuda_registrar.h"
#include "utility/oskar_kernel_macros.h"
#include "utility/oskar_vector_types.h"

/* Kernels */

#define Real float
#define Real2 float2
#define Real4c float4c
#include "mem/src/oskar_mem.cl"
#include "mem/src/oskar_mem_gpu.cl"
#undef Real
#undef Real2
#undef Real4c

#define Real double
#define Real2 double2
#define Real4c double4c
#include "mem/src/oskar_mem.cl"
#include "mem/src/oskar_mem_gpu.cl"
#undef Real
#undef Real2
#undef Real4c
