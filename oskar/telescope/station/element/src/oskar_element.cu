/* Copyright (c) 2018-2021, The OSKAR Developers. See LICENSE file. */

#include "math/oskar_cmath.h"
#include "math/define_legendre_polynomial.h"
#include "math/define_multiply.h"
#include "telescope/station/element/define_apply_element_taper_cosine.h"
#include "telescope/station/element/define_apply_element_taper_gaussian.h"
#include "telescope/station/element/define_evaluate_dipole_pattern.h"
/*#include "telescope/station/element/define_evaluate_geometric_dipole_pattern.h"*/
#include "telescope/station/element/define_evaluate_spherical_wave.h"
#include "utility/oskar_cuda_registrar.h"
#include "utility/oskar_kernel_macros.h"
#include "utility/oskar_vector_types.h"

/* Kernels */

#define Real float
#define Real2 float2
#define Real4c float4c
#include "telescope/station/element/src/oskar_element.cl"
#undef Real
#undef Real2
#undef Real4c

#define Real double
#define Real2 double2
#define Real4c double4c
#include "telescope/station/element/src/oskar_element.cl"
#undef Real
#undef Real2
#undef Real4c
