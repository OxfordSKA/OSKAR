/*
 * Copyright (c) 2013-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <string.h>

#include "math/oskar_cmath.h"
#include "math/oskar_find_closest_match.h"
#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station_accessors.h"


#ifdef __cplusplus
extern "C" {
#endif


/* Data common to all station types. */

int oskar_station_unique_id(const oskar_Station* model)
{
    return model ? model->unique_id : 0;
}

int oskar_station_precision(const oskar_Station* model)
{
    return model ? model->precision : 0;
}

int oskar_station_mem_location(const oskar_Station* model)
{
    return model ? model->mem_location : 0;
}

const char* oskar_station_name(const oskar_Station* model)
{
    return oskar_mem_char_const(model->name);
}

int oskar_station_type(const oskar_Station* model)
{
    return model ? model->station_type : 0;
}

int oskar_station_normalise_final_beam(const oskar_Station* model)
{
    return model ? model->normalise_final_beam : 0;
}

double oskar_station_lon_rad(const oskar_Station* model)
{
    return model ? model->lon_rad : 0.0;
}

double oskar_station_lat_rad(const oskar_Station* model)
{
    return model ? model->lat_rad : 0.0;
}

double oskar_station_alt_metres(const oskar_Station* model)
{
    return model ? model->alt_metres : 0.0;
}

double oskar_station_offset_ecef_x(const oskar_Station* model)
{
    return model ? model->offset_ecef[0] : 0.0;
}

double oskar_station_offset_ecef_y(const oskar_Station* model)
{
    return model ? model->offset_ecef[1] : 0.0;
}

double oskar_station_offset_ecef_z(const oskar_Station* model)
{
    return model ? model->offset_ecef[2] : 0.0;
}

double oskar_station_polar_motion_x_rad(const oskar_Station* model)
{
    return model ? model->pm_x_rad : 0.0;
}

double oskar_station_polar_motion_y_rad(const oskar_Station* model)
{
    return model ? model->pm_y_rad : 0.0;
}

double oskar_station_beam_lon_rad(const oskar_Station* model)
{
    return model ? model->beam_lon_rad : 0.0;
}

double oskar_station_beam_lat_rad(const oskar_Station* model)
{
    return model ? model->beam_lat_rad : 0.0;
}

int oskar_station_beam_coord_type(const oskar_Station* model)
{
    return model ? model->beam_coord_type : 0;
}

const double* oskar_station_magnetic_field(const oskar_Station* model)
{
    return model ? model->magnetic_field : 0;
}

oskar_Mem* oskar_station_noise_freq_hz(oskar_Station* model)
{
    return model ? model->noise_freq_hz : 0;
}

const oskar_Mem* oskar_station_noise_freq_hz_const(const oskar_Station* model)
{
    return model ? model->noise_freq_hz : 0;
}

oskar_Mem* oskar_station_noise_rms_jy(oskar_Station* model)
{
    return model ? model->noise_rms_jy : 0;
}

const oskar_Mem* oskar_station_noise_rms_jy_const(const oskar_Station* model)
{
    return model ? model->noise_rms_jy : 0;
}

/* Data used only for Gaussian beam stations. */

double oskar_station_gaussian_beam_fwhm_rad(const oskar_Station* model,
        int feed, int dim
)
{
    return model ? model->gaussian_beam_fwhm_rad[feed][dim] : 0.0;
}

double oskar_station_gaussian_beam_reference_freq_hz(const oskar_Station* model)
{
    return model ? model->gaussian_beam_reference_freq_hz : 0.0;
}

double oskar_station_gaussian_beam_sincos_sq_pa(
        const oskar_Station* model, int feed, int dim
)
{
    return model ? model->gaussian_beam_sincos_sq_pa[feed][dim] : 0.0;
}

double oskar_station_gaussian_beam_sin_2_pa(
        const oskar_Station* model, int feed
)
{
    return model ? model->gaussian_beam_sin_2_pa[feed] : 0.0;
}

int oskar_station_gaussian_beam_use_ellipse(const oskar_Station* model)
{
    return model ? model->gaussian_beam_use_ellipse : 0;
}

/* Data used only for aperture array stations. */

int oskar_station_identical_children(const oskar_Station* model)
{
    return model ? model->identical_children : 0;
}

int oskar_station_num_elements(const oskar_Station* model)
{
    return model ? model->num_elements : 0;
}

int oskar_station_num_element_types(const oskar_Station* model)
{
    return model ? model->num_element_types : 0;
}

int oskar_station_normalise_array_pattern(const oskar_Station* model)
{
    return model ? model->normalise_array_pattern : 0;
}

int oskar_station_normalise_element_pattern(const oskar_Station* model)
{
    return model ? model->normalise_element_pattern : 0;
}

int oskar_station_enable_array_pattern(const oskar_Station* model)
{
    return model ? model->enable_array_pattern : 0;
}

int oskar_station_common_element_orientation(const oskar_Station* model)
{
    return model ? model->common_element_orientation : 0;
}

int oskar_station_common_pol_beams(const oskar_Station* model)
{
    return model ? model->common_pol_beams : 0;
}

int oskar_station_array_is_3d(const oskar_Station* model)
{
    return model ? model->array_is_3d : 0;
}

int oskar_station_apply_element_errors(const oskar_Station* model)
{
    return model ? model->apply_element_errors : 0;
}

int oskar_station_apply_element_weight(const oskar_Station* model)
{
    return model ? model->apply_element_weight : 0;
}

unsigned int oskar_station_seed_time_variable_errors(const oskar_Station* model)
{
    return model ? model->seed_time_variable_errors : 0u;
}

int oskar_station_swap_xy(const oskar_Station* model)
{
    return model ? model->swap_xy : 0;
}

double oskar_station_element_euler_index_rad(
        const oskar_Station* model, int feed, int dim, int index)
{
    const oskar_Mem* ptr = 0;
    if (!model || feed > 1 || dim > 2) return 0.0;
    ptr = model->element_euler_cpu[feed][dim];
    if (!ptr) ptr = model->element_euler_cpu[0][dim];
    return ptr ? ((const double*) oskar_mem_void_const(ptr))[index] : 0.0;
}

oskar_Mem* oskar_station_element_euler_rad(
        oskar_Station* model, int feed, int dim)
{
    oskar_Mem* ptr = 0;
    if (!model || feed > 1 || dim > 2) return 0;
    ptr = model->element_euler_cpu[feed][dim];
    return ptr ? ptr : model->element_euler_cpu[0][dim];
}

const oskar_Mem* oskar_station_element_euler_rad_const(
        const oskar_Station* model, int feed, int dim)
{
    const oskar_Mem* ptr = 0;
    if (!model || feed > 1 || dim > 2) return 0;
    ptr = model->element_euler_cpu[feed][dim];
    return ptr ? ptr : model->element_euler_cpu[0][dim];
}

oskar_Mem* oskar_station_element_true_enu_metres(
        oskar_Station* model, int feed, int dim)
{
    oskar_Mem* ptr = 0;
    if (!model || feed > 1 || dim > 2) return 0;
    ptr = model->element_true_enu_metres[feed][dim];
    return ptr ? ptr : model->element_true_enu_metres[0][dim];
}

const oskar_Mem* oskar_station_element_true_enu_metres_const(
        const oskar_Station* model, int feed, int dim)
{
    const oskar_Mem* ptr = 0;
    if (!model || feed > 1 || dim > 2) return 0;
    ptr = model->element_true_enu_metres[feed][dim];
    return ptr ? ptr : model->element_true_enu_metres[0][dim];
}

oskar_Mem* oskar_station_element_measured_enu_metres(
        oskar_Station* model, int feed, int dim)
{
    oskar_Mem* ptr = 0;
    if (!model || feed > 1 || dim > 2) return 0;
    ptr = model->element_measured_enu_metres[feed][dim];
    return ptr ? ptr : model->element_measured_enu_metres[0][dim];
}

const oskar_Mem* oskar_station_element_measured_enu_metres_const(
        const oskar_Station* model, int feed, int dim)
{
    const oskar_Mem* ptr = 0;
    if (!model || feed > 1 || dim > 2) return 0;
    ptr = model->element_measured_enu_metres[feed][dim];
    return ptr ? ptr : model->element_measured_enu_metres[0][dim];
}

oskar_Mem* oskar_station_element_cable_length_error_metres(
        oskar_Station* model, int feed)
{
    oskar_Mem* ptr = 0;
    if (!model || feed > 1) return 0;
    ptr = model->element_cable_length_error[feed];
    return ptr ? ptr : model->element_cable_length_error[0];
}

const oskar_Mem* oskar_station_element_cable_length_error_metres_const(
        const oskar_Station* model, int feed)
{
    const oskar_Mem* ptr = 0;
    if (!model || feed > 1) return 0;
    ptr = model->element_cable_length_error[feed];
    return ptr ? ptr : model->element_cable_length_error[0];
}

oskar_Mem* oskar_station_element_gain(oskar_Station* model, int feed)
{
    oskar_Mem* ptr = 0;
    if (!model || feed > 1) return 0;
    ptr = model->element_gain[feed];
    return ptr ? ptr : model->element_gain[0];
}

const oskar_Mem* oskar_station_element_gain_const(
        const oskar_Station* model, int feed)
{
    const oskar_Mem* ptr = 0;
    if (!model || feed > 1) return 0;
    ptr = model->element_gain[feed];
    return ptr ? ptr : model->element_gain[0];
}

oskar_Mem* oskar_station_element_gain_error(oskar_Station* model, int feed)
{
    oskar_Mem* ptr = 0;
    if (!model || feed > 1) return 0;
    ptr = model->element_gain_error[feed];
    return ptr ? ptr : model->element_gain_error[0];
}

const oskar_Mem* oskar_station_element_gain_error_const(
        const oskar_Station* model, int feed)
{
    const oskar_Mem* ptr = 0;
    if (!model || feed > 1) return 0;
    ptr = model->element_gain_error[feed];
    return ptr ? ptr : model->element_gain_error[0];
}

oskar_Mem* oskar_station_element_phase_offset_rad(
        oskar_Station* model, int feed)
{
    oskar_Mem* ptr = 0;
    if (!model || feed > 1) return 0;
    ptr = model->element_phase_offset_rad[feed];
    return ptr ? ptr : model->element_phase_offset_rad[0];
}

const oskar_Mem* oskar_station_element_phase_offset_rad_const(
        const oskar_Station* model, int feed)
{
    const oskar_Mem* ptr = 0;
    if (!model || feed > 1) return 0;
    ptr = model->element_phase_offset_rad[feed];
    return ptr ? ptr : model->element_phase_offset_rad[0];
}

oskar_Mem* oskar_station_element_phase_error_rad(
        oskar_Station* model, int feed)
{
    oskar_Mem* ptr = 0;
    if (!model || feed > 1) return 0;
    ptr = model->element_phase_error_rad[feed];
    return ptr ? ptr : model->element_phase_error_rad[0];
}

const oskar_Mem* oskar_station_element_phase_error_rad_const(
        const oskar_Station* model, int feed)
{
    const oskar_Mem* ptr = 0;
    if (!model || feed > 1) return 0;
    ptr = model->element_phase_error_rad[feed];
    return ptr ? ptr : model->element_phase_error_rad[0];
}

oskar_Mem* oskar_station_element_weight(oskar_Station* model, int feed)
{
    oskar_Mem* ptr = 0;
    if (!model || feed > 1) return 0;
    ptr = model->element_weight[feed];
    return ptr ? ptr : model->element_weight[0];
}

const oskar_Mem* oskar_station_element_weight_const(
        const oskar_Station* model, int feed)
{
    const oskar_Mem* ptr = 0;
    if (!model || feed > 1) return 0;
    ptr = model->element_weight[feed];
    return ptr ? ptr : model->element_weight[0];
}

oskar_Mem* oskar_station_element_types(oskar_Station* model)
{
    return model ? model->element_types : 0;
}

const oskar_Mem* oskar_station_element_types_const(const oskar_Station* model)
{
    return model ? model->element_types : 0;
}

const int* oskar_station_element_types_cpu_const(const oskar_Station* model)
{
    if (!model) return 0;
    return (const int*) oskar_mem_void_const(model->element_types_cpu);
}

const char* oskar_station_element_mount_types_const(const oskar_Station* model)
{
    if (!model) return 0;
    return oskar_mem_char_const(model->element_mount_types_cpu);
}

int oskar_station_has_child(const oskar_Station* model)
{
    return model ? (model->child ? 1 : 0) : 0;
}

oskar_Station* oskar_station_child(oskar_Station* model, int i)
{
    return model ? model->child[i] : 0;
}

const oskar_Station* oskar_station_child_const(const oskar_Station* model,
        int i)
{
    return model ? model->child[i] : 0;
}

int oskar_station_has_element(const oskar_Station* model)
{
    return model ? (model->element ? 1 : 0) : 0;
}

oskar_Element* oskar_station_element(oskar_Station* model,
        int element_type_index)
{
    return model ? model->element[element_type_index] : 0;
}

const oskar_Element* oskar_station_element_const(const oskar_Station* model,
        int element_type_index)
{
    return model ? model->element[element_type_index] : 0;
}

int oskar_station_num_permitted_beams(const oskar_Station* model)
{
    return model ? model->num_permitted_beams : 0;
}

const oskar_Mem* oskar_station_permitted_beam_az_rad_const(
        const oskar_Station* model)
{
    return model ? model->permitted_beam_az_rad : 0;
}

const oskar_Mem* oskar_station_permitted_beam_el_rad_const(
        const oskar_Station* model)
{
    return model ? model->permitted_beam_el_rad : 0;
}

oskar_Gains* oskar_station_gains(oskar_Station* model)
{
    return model ? model->gains : 0;
}

const oskar_Gains* oskar_station_gains_const(const oskar_Station* model)
{
    return model ? model->gains : 0;
}

oskar_Harp* oskar_station_harp_data(oskar_Station* model,
        double freq_hz)
{
    int index = 0, status = 0;
    if (!model || !model->harp_data) return 0;
    index = oskar_find_closest_match(freq_hz, model->harp_freq_cpu, &status);
    return model->harp_data[index];
}

const oskar_Harp* oskar_station_harp_data_const(const oskar_Station* model,
        double freq_hz)
{
    int index = 0, status = 0;
    if (!model || !model->harp_data) return 0;
    index = oskar_find_closest_match(freq_hz, model->harp_freq_cpu, &status);
    return model->harp_data[index];
}

double oskar_station_virtual_antenna_angle_rad(const oskar_Station* model)
{
    return (!model) ? 0.0 : model->virtual_antenna_angle_rad;
}

/* Setters. */

void oskar_station_set_unique_ids(oskar_Station* model, int* counter)
{
    if (!model) return;
    model->unique_id = (*counter)++;
    if (model->child)
    {
        int i = 0;
        for (i = 0; i < model->num_elements; ++i)
        {
            oskar_station_set_unique_ids(model->child[i], counter);
        }
    }
}

void oskar_station_set_name(
        oskar_Station* model,
        const char* name,
        int* status
)
{
    const size_t len = 1 + strlen(name);
    oskar_mem_realloc(model->name, len, status);
    if (!*status)
    {
        memcpy(oskar_mem_void(model->name), name, len);
    }
}

void oskar_station_set_station_type(oskar_Station* model, int type)
{
    if (!model) return;
    model->station_type = type;
}

void oskar_station_set_normalise_final_beam(oskar_Station* model, int value)
{
    if (!model) return;
    model->normalise_final_beam = value;
}

void oskar_station_set_position(oskar_Station* model,
        double longitude_rad, double latitude_rad, double altitude_m,
        double offset_ecef_x, double offset_ecef_y, double offset_ecef_z)
{
    if (!model) return;
    model->lon_rad = longitude_rad;
    model->lat_rad = latitude_rad;
    model->alt_metres = altitude_m;
    model->offset_ecef[0] = offset_ecef_x;
    model->offset_ecef[1] = offset_ecef_y;
    model->offset_ecef[2] = offset_ecef_z;
}

void oskar_station_set_polar_motion(oskar_Station* model,
        double pm_x_rad, double pm_y_rad)
{
    int i = 0;
    if (!model) return;
    model->pm_x_rad = pm_x_rad;
    model->pm_y_rad = pm_y_rad;

    /* Set recursively for all child stations. */
    if (oskar_station_has_child(model))
    {
        for (i = 0; i < model->num_elements; ++i)
        {
            oskar_station_set_polar_motion(model->child[i], pm_x_rad, pm_y_rad);
        }
    }
}

void oskar_station_set_phase_centre(oskar_Station* model,
        int coord_type, double longitude_rad, double latitude_rad)
{
    if (!model) return;
    model->beam_coord_type = coord_type;
    model->beam_lon_rad = longitude_rad;
    model->beam_lat_rad = latitude_rad;
}

void oskar_station_set_gaussian_beam_values(oskar_Station* model,
        double fwhm_maj_rad, double fwhm_min_rad, double pa_rad,
        double ref_freq_hz, int feed)
{
    if (!model) return;
    model->gaussian_beam_use_ellipse = (fwhm_min_rad > 0.0) ? 1 : 0;
    model->gaussian_beam_fwhm_rad[feed][0] = fwhm_maj_rad;
    model->gaussian_beam_fwhm_rad[feed][1] = fwhm_min_rad;
    model->gaussian_beam_sin_2_pa[feed] = sin(2.0 * pa_rad);
    model->gaussian_beam_sincos_sq_pa[feed][0] = sin(pa_rad) * sin(pa_rad);
    model->gaussian_beam_sincos_sq_pa[feed][1] = cos(pa_rad) * cos(pa_rad);
    model->gaussian_beam_reference_freq_hz = ref_freq_hz;
}

void oskar_station_set_normalise_array_pattern(oskar_Station* model, int value)
{
    if (!model) return;
    model->normalise_array_pattern = value;
}

void oskar_station_set_normalise_element_pattern(oskar_Station* model, int value)
{
    if (!model) return;
    model->normalise_element_pattern = value;
}

void oskar_station_set_enable_array_pattern(oskar_Station* model, int value)
{
    if (!model) return;
    model->enable_array_pattern = value;
}

void oskar_station_set_seed_time_variable_errors(oskar_Station* model,
        unsigned int value)
{
    if (!model) return;
    model->seed_time_variable_errors = value;
}

void oskar_station_set_swap_xy(oskar_Station* model, int value)
{
    if (!model) return;
    model->swap_xy = value;
}

void oskar_station_set_virtual_antenna_angle(oskar_Station* model,
        double value_deg)
{
    if (!model) return;
    model->virtual_antenna_angle_rad = value_deg * M_PI / 180.0;
}

#ifdef __cplusplus
}
#endif
