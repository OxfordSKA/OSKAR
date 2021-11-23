/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "splines/private_splines.h"
#include "splines/oskar_splines.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_splines_precision(const oskar_Splines* data)
{
    return data->precision;
}

int oskar_splines_mem_location(const oskar_Splines* data)
{
    return data->mem_location;
}

int oskar_splines_have_coeffs(const oskar_Splines* data)
{
    return data &&
            (data->num_knots_x_theta > 0) &&
            (data->num_knots_y_phi > 0);
}

int oskar_splines_num_knots_x_theta(const oskar_Splines* data)
{
    return data ? data->num_knots_x_theta : 0;
}

int oskar_splines_num_knots_y_phi(const oskar_Splines* data)
{
    return data ? data->num_knots_y_phi : 0;
}

oskar_Mem* oskar_splines_knots_x(oskar_Splines* data)
{
    return data->knots_x_theta;
}

const oskar_Mem* oskar_splines_knots_x_theta_const(const oskar_Splines* data)
{
    return data->knots_x_theta;
}

oskar_Mem* oskar_splines_knots_y(oskar_Splines* data)
{
    return data->knots_y_phi;
}

const oskar_Mem* oskar_splines_knots_y_phi_const(const oskar_Splines* data)
{
    return data->knots_y_phi;
}

oskar_Mem* oskar_splines_coeff(oskar_Splines* data)
{
    return data->coeff;
}

const oskar_Mem* oskar_splines_coeff_const(const oskar_Splines* data)
{
    return data->coeff;
}

double oskar_splines_smoothing_factor(const oskar_Splines* data)
{
    return data->smoothing_factor;
}

void oskar_splines_copy(oskar_Splines* dst, const oskar_Splines* src,
        int* status)
{
    if (*status || !dst || !src) return;
    dst->precision = src->precision;
    dst->smoothing_factor = src->smoothing_factor;
    dst->num_knots_x_theta = src->num_knots_x_theta;
    dst->num_knots_y_phi = src->num_knots_y_phi;
    if (src->num_knots_x_theta > 0)
    {
        oskar_mem_copy(dst->knots_x_theta, src->knots_x_theta, status);
    }
    if (src->num_knots_y_phi > 0)
    {
        oskar_mem_copy(dst->knots_y_phi, src->knots_y_phi, status);
    }
    if ((src->num_knots_x_theta) > 0 || (src->num_knots_y_phi > 0))
    {
        oskar_mem_copy(dst->coeff, src->coeff, status);
    }
}

oskar_Splines* oskar_splines_create(int precision, int location, int* status)
{
    oskar_Splines* data = (oskar_Splines*) calloc(1, sizeof(oskar_Splines));
    if (!data)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        return 0;
    }
    data->precision = precision;
    data->mem_location = location;
    data->knots_x_theta = oskar_mem_create(precision, location, 0, status);
    data->knots_y_phi = oskar_mem_create(precision, location, 0, status);
    data->coeff = oskar_mem_create(precision, location, 0, status);
    return data;
}

void oskar_splines_free(oskar_Splines* data, int* status)
{
    if (!data) return;
    oskar_mem_free(data->knots_x_theta, status);
    oskar_mem_free(data->knots_y_phi, status);
    oskar_mem_free(data->coeff, status);
    free(data);
}

#ifdef __cplusplus
}
#endif
