/* Copyright (c) 2018-2019, The University of Oxford. See LICENSE file. */

#include "imager/define_grid_correction.h"
#include "imager/define_grid_tile_grid.h"
#include "imager/define_grid_tile_utils.h"
#include "imager/define_imager_generate_w_phase_screen.h"
#include "utility/oskar_cuda_registrar.h"
#include "utility/oskar_kernel_macros.h"
#include "utility/oskar_vector_types.h"

/* Kernels */
#define PREFER_DOUBLE double

#define Real float
#define Real2 float2
#include "imager/src/oskar_imager.cl"
#include "imager/src/oskar_imager_gpu.cl"
#undef Real
#undef Real2

#define Real double
#define Real2 double2
#include "imager/src/oskar_imager.cl"
#include "imager/src/oskar_imager_gpu.cl"
#undef Real
#undef Real2
