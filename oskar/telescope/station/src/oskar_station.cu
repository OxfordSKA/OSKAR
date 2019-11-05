/* Copyright (c) 2018-2019, The University of Oxford. See LICENSE file. */

#include "telescope/station/define_blank_below_horizon.h"
#include "telescope/station/define_evaluate_element_weights_dft.h"
#include "telescope/station/define_evaluate_element_weights_errors.h"
#include "telescope/station/define_evaluate_tec_screen.h"
#include "telescope/station/define_evaluate_vla_beam_pbcor.h"
#include "utility/oskar_cuda_registrar.h"
#include "utility/oskar_kernel_macros.h"
#include "utility/oskar_vector_types.h"

/* Kernels */

#define Real float
#define Real2 float2
#define Real4c float4c
#include "telescope/station/src/oskar_station.cl"
#undef Real
#undef Real2
#undef Real4c

#define Real double
#define Real2 double2
#define Real4c double4c
#include "telescope/station/src/oskar_station.cl"
#undef Real
#undef Real2
#undef Real4c
