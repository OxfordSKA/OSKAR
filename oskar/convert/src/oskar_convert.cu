/* Copyright (c) 2018-2021, The OSKAR Developers. See LICENSE file. */

#include "convert/define_convert_az_el_to_enu_directions.h"
#include "convert/define_convert_apparent_ra_dec_to_enu_directions.h"
#include "convert/define_convert_cirs_relative_directions_to_enu_directions.h"
#include "convert/define_convert_ecef_to_station_uvw.h"
#include "convert/define_convert_enu_directions_to_az_el.h"
/*#include "convert/define_convert_enu_directions_to_cirs_relative_directions.h"*/
#include "convert/define_convert_enu_directions_to_local_tangent_plane.h"
#include "convert/define_convert_enu_directions_to_relative_directions.h"
#include "convert/define_convert_enu_directions_to_theta_phi.h"
#include "convert/define_convert_lon_lat_to_relative_directions.h"
#include "convert/define_convert_ludwig3_to_theta_phi_components.h"
#include "convert/define_convert_relative_directions_to_enu_directions.h"
#include "convert/define_convert_relative_directions_to_lon_lat.h"
#include "convert/define_convert_station_uvw_to_baseline_uvw.h"
#include "convert/define_convert_theta_phi_to_ludwig3_components.h"
#include "utility/oskar_cuda_registrar.h"
#include "utility/oskar_kernel_macros.h"
#include "utility/oskar_vector_types.h"

/* Kernels */

#define Real float
#define Real2 float2
#define Real4c float4c
#include "convert/src/oskar_convert.cl"
#undef Real
#undef Real2
#undef Real4c

#define Real double
#define Real2 double2
#define Real4c double4c
#include "convert/src/oskar_convert.cl"
#undef Real
#undef Real2
#undef Real4c
