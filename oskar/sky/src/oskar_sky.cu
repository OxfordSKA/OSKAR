/* Copyright (c) 2018, The University of Oxford. See LICENSE file. */

#include "sky/define_sky_copy_source_data.h"
#include "sky/define_sky_scale_flux_with_frequency.h"
#include "sky/define_update_horizon_mask.h"
#include "utility/oskar_cuda_registrar.h"
#include "utility/oskar_kernel_macros.h"
#include "utility/oskar_vector_types.h"

/* Kernels */

#define Real float
#include "sky/src/oskar_sky.cl"
#undef Real

#define Real double
#include "sky/src/oskar_sky.cl"
#undef Real
