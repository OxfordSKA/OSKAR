/*
 * Copyright (c) 2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdlib.h>

#include "harp/oskar_harp.h"
#include "harp/private_harp.h"
#include "log/oskar_log.h"
#include "utility/oskar_thread.h"

#ifdef OSKAR_HAVE_HARP
#include "harp_beam.h"
#endif

oskar_Harp* oskar_harp_create(int precision)
{
    oskar_Harp* h = (oskar_Harp*) calloc(1, sizeof(oskar_Harp));
    h->precision = precision;
    return h;
}

/* NOLINTNEXTLINE(readability-non-const-parameter) */
oskar_Harp* oskar_harp_create_copy(const oskar_Harp* other, int* status)
{
    (void)status;
    oskar_Harp* h = (oskar_Harp*) calloc(1, sizeof(oskar_Harp));
    h->precision = other->precision;
    h->num_antennas = other->num_antennas;
    h->num_mbf = other->num_mbf;
    h->max_order = other->max_order;
    h->freq = other->freq;
    h->hdf5_file = other->hdf5_file;
    oskar_hdf5_ref_inc(h->hdf5_file);
    h->alpha_te = other->alpha_te;
    oskar_mem_ref_inc(h->alpha_te);
    h->alpha_tm = other->alpha_tm;
    oskar_mem_ref_inc(h->alpha_tm);
    h->coeffs_pola = other->coeffs_pola;
    oskar_mem_ref_inc(h->coeffs_pola);
    h->coeffs_polb = other->coeffs_polb;
    oskar_mem_ref_inc(h->coeffs_polb);
    return h;
}

const oskar_Mem* oskar_harp_coeffs(const oskar_Harp* h, int feed)
{
    return feed == 0 ? h->coeffs_pola : h->coeffs_polb;
}

void oskar_harp_evaluate_smodes(
        const oskar_Harp* h,
        int num_dir,
        const oskar_Mem* theta,
        const oskar_Mem* phi,
        oskar_Mem* poly,
        oskar_Mem* ee,
        oskar_Mem* qq,
        oskar_Mem* dd,
        oskar_Mem* pth,
        oskar_Mem* pph,
        int* status)
{
    if (*status) return;
#ifdef OSKAR_HAVE_HARP
    const int max_order = h->max_order;
    oskar_mem_ensure(poly, num_dir * max_order * (max_order + 1), status);
    oskar_mem_ensure(ee, num_dir * (2 * max_order + 1), status);
    oskar_mem_ensure(qq, num_dir * max_order * (2 * max_order + 1), status);
    oskar_mem_ensure(dd, num_dir * max_order * (2 * max_order + 1), status);
    oskar_mem_ensure(pth, num_dir * h->num_mbf, status);
    oskar_mem_ensure(pph, num_dir * h->num_mbf, status);
    if (*status) return;
    const int precision = oskar_mem_precision(pth);
    const int location = oskar_mem_location(pth);
    oskar_Mem *gpu_te = 0, *gpu_tm = 0;
    oskar_Mem *ptr_te = h->alpha_te, *ptr_tm = h->alpha_tm;
    if (oskar_mem_location(h->alpha_te) != location)
    {
        gpu_te = oskar_mem_create_copy(ptr_te, location, status);
        gpu_tm = oskar_mem_create_copy(ptr_tm, location, status);
        ptr_te = gpu_te;
        ptr_tm = gpu_tm;
    }
    if (location == OSKAR_CPU)
    {
        if (precision == OSKAR_DOUBLE)
        {
            harp_precompute_smodes_mbf_double(
                    h->max_order,
                    num_dir,
                    oskar_mem_double_const(theta, status),
                    oskar_mem_double_const(phi, status),
                    oskar_mem_double(poly, status),
                    oskar_mem_double2(ee, status),
                    oskar_mem_double2(qq, status),
                    oskar_mem_double2(dd, status));
            harp_reconstruct_smodes_mbf_double(
                    h->max_order,
                    num_dir,
                    h->num_mbf,
                    oskar_mem_double2_const(ptr_te, status),
                    oskar_mem_double2_const(ptr_tm, status),
                    oskar_mem_double2_const(qq, status),
                    oskar_mem_double2_const(dd, status),
                    oskar_mem_double2(pth, status),
                    oskar_mem_double2(pph, status));
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (precision == OSKAR_DOUBLE)
        {
            harp_precompute_smodes_mbf_cuda_double(
                    h->max_order,
                    num_dir,
                    oskar_mem_double_const(theta, status),
                    oskar_mem_double_const(phi, status),
                    oskar_mem_double(poly, status),
                    oskar_mem_double2(ee, status),
                    oskar_mem_double2(qq, status),
                    oskar_mem_double2(dd, status));
            harp_reconstruct_smodes_mbf_cuda_double(
                    h->max_order,
                    num_dir,
                    h->num_mbf,
                    oskar_mem_double2_const(ptr_te, status),
                    oskar_mem_double2_const(ptr_tm, status),
                    oskar_mem_double2_const(qq, status),
                    oskar_mem_double2_const(dd, status),
                    oskar_mem_double2(pth, status),
                    oskar_mem_double2(pph, status));
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
#endif
    }
    else
    {
        *status = OSKAR_ERR_BAD_LOCATION;
    }
    oskar_mem_free(gpu_te, status);
    oskar_mem_free(gpu_tm, status);
#else
    (void)h;
    (void)num_dir;
    (void)theta;
    (void)phi;
    (void)poly;
    (void)ee;
    (void)qq;
    (void)dd;
    (void)pth;
    (void)pph;
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    oskar_log_error(0, "OSKAR was compiled without HARP support");
#endif
}

void oskar_harp_evaluate_station_beam(
        const oskar_Harp* h,
        int num_dir,
        const oskar_Mem* theta,
        const oskar_Mem* phi,
        double frequency_hz,
        int feed,
        int num_antennas,
        const oskar_Mem* antenna_x,
        const oskar_Mem* antenna_y,
        const oskar_Mem* antenna_z,
        const oskar_Mem* weights,
        const oskar_Mem* pth,
        const oskar_Mem* pph,
        oskar_Mem* phase_fac,
        oskar_Mem* beam_coeffs,
        int offset_out,
        oskar_Mem* beam,
        int* status)
{
    if (*status) return;
#ifdef OSKAR_HAVE_HARP
    oskar_mem_ensure(phase_fac, num_dir * num_antennas, status);
    oskar_mem_ensure(beam_coeffs, h->num_mbf * num_antennas, status);
    oskar_mem_ensure(beam, num_dir, status);
    if (*status) return;
    const int precision = oskar_mem_precision(beam);
    const int location = oskar_mem_location(beam);
    const int stride = 4;
    const int offset_out_cplx = offset_out * stride + 2 * feed;
    oskar_Mem *gpu_coeffs = 0;
    oskar_Mem *ptr_coeffs = (feed == 0 ? h->coeffs_pola : h->coeffs_polb);
    if (oskar_mem_location(ptr_coeffs) != location)
    {
        ptr_coeffs = gpu_coeffs = oskar_mem_create_copy(
                ptr_coeffs, location, status);
    }
    if (location == OSKAR_CPU)
    {
        if (precision == OSKAR_DOUBLE)
        {
            harp_evaluate_beam_coeffs_double(
                    h->num_mbf,
                    num_antennas,
                    oskar_mem_double2_const(weights, status),
                    oskar_mem_double2_const(ptr_coeffs, status),
                    oskar_mem_double2(beam_coeffs, status));
            harp_evaluate_phase_fac_double(
                    num_dir,
                    num_antennas,
                    frequency_hz,
                    oskar_mem_double_const(theta, status),
                    oskar_mem_double_const(phi, status),
                    oskar_mem_double_const(antenna_x, status),
                    oskar_mem_double_const(antenna_y, status),
                    oskar_mem_double_const(antenna_z, status),
                    oskar_mem_double2(phase_fac, status));
            harp_assemble_station_beam_double(
                    h->num_mbf,
                    num_antennas,
                    num_dir,
                    oskar_mem_double2_const(beam_coeffs, status),
                    oskar_mem_double2_const(pth, status),
                    oskar_mem_double2_const(pph, status),
                    oskar_mem_double2_const(phase_fac, status),
                    stride,
                    offset_out_cplx,
                    offset_out_cplx + 1,
                    oskar_mem_double2(beam, status),
                    oskar_mem_double2(beam, status));
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (precision == OSKAR_DOUBLE)
        {
            harp_evaluate_beam_coeffs_cuda_double(
                    h->num_mbf,
                    num_antennas,
                    oskar_mem_double2_const(weights, status),
                    oskar_mem_double2_const(ptr_coeffs, status),
                    oskar_mem_double2(beam_coeffs, status));
            harp_evaluate_phase_fac_cuda_double(
                    num_dir,
                    num_antennas,
                    frequency_hz,
                    oskar_mem_double_const(theta, status),
                    oskar_mem_double_const(phi, status),
                    oskar_mem_double_const(antenna_x, status),
                    oskar_mem_double_const(antenna_y, status),
                    oskar_mem_double_const(antenna_z, status),
                    oskar_mem_double2(phase_fac, status));
            harp_assemble_station_beam_cuda_double(
                    h->num_mbf,
                    num_antennas,
                    num_dir,
                    oskar_mem_double2_const(beam_coeffs, status),
                    oskar_mem_double2_const(pth, status),
                    oskar_mem_double2_const(pph, status),
                    oskar_mem_double2_const(phase_fac, status),
                    stride,
                    offset_out_cplx,
                    offset_out_cplx + 1,
                    oskar_mem_double2(beam, status),
                    oskar_mem_double2(beam, status));
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
#endif
    }
    else
    {
        *status = OSKAR_ERR_BAD_LOCATION;
    }
    oskar_mem_free(gpu_coeffs, status);
#else
    (void)h;
    (void)num_dir;
    (void)theta;
    (void)phi;
    (void)frequency_hz;
    (void)feed;
    (void)num_antennas;
    (void)antenna_x;
    (void)antenna_y;
    (void)antenna_z;
    (void)weights;
    (void)pth;
    (void)pph;
    (void)phase_fac;
    (void)beam_coeffs;
    (void)offset_out;
    (void)beam;
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    oskar_log_error(0, "OSKAR was compiled without HARP support");
#endif
}

void oskar_harp_evaluate_element_beams(
        const oskar_Harp* h,
        int num_dir,
        const oskar_Mem* theta,
        const oskar_Mem* phi,
        double frequency_hz,
        int feed,
        int num_antennas,
        const oskar_Mem* antenna_x,
        const oskar_Mem* antenna_y,
        const oskar_Mem* antenna_z,
        const oskar_Mem* coeffs,
        const oskar_Mem* pth,
        const oskar_Mem* pph,
        oskar_Mem* phase_fac,
        int offset_out,
        oskar_Mem* beam,
        int* status)
{
    if (*status) return;
#ifdef OSKAR_HAVE_HARP
    oskar_mem_ensure(phase_fac, num_dir * num_antennas, status);
    oskar_mem_ensure(beam, num_dir, status);
    if (*status) return;
    const int precision = oskar_mem_precision(beam);
    const int location = oskar_mem_location(beam);
    const int stride = 4;
    const int offset_out_cplx = offset_out * stride + 2 * feed;
    if (location == OSKAR_CPU)
    {
        if (precision == OSKAR_DOUBLE)
        {
            harp_evaluate_phase_fac_double(
                    num_dir,
                    num_antennas,
                    frequency_hz,
                    oskar_mem_double_const(theta, status),
                    oskar_mem_double_const(phi, status),
                    oskar_mem_double_const(antenna_x, status),
                    oskar_mem_double_const(antenna_y, status),
                    oskar_mem_double_const(antenna_z, status),
                    oskar_mem_double2(phase_fac, status));
            harp_assemble_element_beams_double(
                    h->num_mbf,
                    num_antennas,
                    num_dir,
                    oskar_mem_double2_const(coeffs, status),
                    oskar_mem_double2_const(pth, status),
                    oskar_mem_double2_const(pph, status),
                    oskar_mem_double2_const(phase_fac, status),
                    stride,
                    offset_out_cplx,
                    offset_out_cplx + 1,
                    oskar_mem_double2(beam, status),
                    oskar_mem_double2(beam, status));
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        if (precision == OSKAR_DOUBLE)
        {
            harp_evaluate_phase_fac_cuda_double(
                    num_dir,
                    num_antennas,
                    frequency_hz,
                    oskar_mem_double_const(theta, status),
                    oskar_mem_double_const(phi, status),
                    oskar_mem_double_const(antenna_x, status),
                    oskar_mem_double_const(antenna_y, status),
                    oskar_mem_double_const(antenna_z, status),
                    oskar_mem_double2(phase_fac, status));
            harp_assemble_element_beams_cuda_double(
                    h->num_mbf,
                    num_antennas,
                    num_dir,
                    oskar_mem_double2_const(coeffs, status),
                    oskar_mem_double2_const(pth, status),
                    oskar_mem_double2_const(pph, status),
                    oskar_mem_double2_const(phase_fac, status),
                    stride,
                    offset_out_cplx,
                    offset_out_cplx + 1,
                    oskar_mem_double2(beam, status),
                    oskar_mem_double2(beam, status));
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
#endif
    }
    else
    {
        *status = OSKAR_ERR_BAD_LOCATION;
    }
#else
    (void)h;
    (void)num_dir;
    (void)theta;
    (void)phi;
    (void)frequency_hz;
    (void)feed;
    (void)num_antennas;
    (void)antenna_x;
    (void)antenna_y;
    (void)antenna_z;
    (void)coeffs;
    (void)pth;
    (void)pph;
    (void)phase_fac;
    (void)offset_out;
    (void)beam;
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    oskar_log_error(0, "OSKAR was compiled without HARP support");
#endif
}

void oskar_harp_free(oskar_Harp* h)
{
    int status = 0;
    if (!h) return;
    oskar_hdf5_close(h->hdf5_file);
    oskar_mem_free(h->alpha_te, &status);
    oskar_mem_free(h->alpha_tm, &status);
    oskar_mem_free(h->coeffs_pola, &status);
    oskar_mem_free(h->coeffs_polb, &status);
    free(h);
}

void oskar_harp_open_hdf5(oskar_Harp* h, const char* path, int* status)
{
    if (*status) return;
    h->hdf5_file = oskar_hdf5_open(path, status);

    /* Load the attributes. */
    h->freq = oskar_hdf5_read_attribute_double(
            h->hdf5_file, "freq", status);
    h->num_antennas = oskar_hdf5_read_attribute_int(
            h->hdf5_file, "num_ant", status);
    h->num_mbf = oskar_hdf5_read_attribute_int(
            h->hdf5_file, "num_mbf", status);
    h->max_order = oskar_hdf5_read_attribute_int(
            h->hdf5_file, "max_order", status);

    /* Load the data. */
    oskar_Mem* alpha_te = oskar_hdf5_read_dataset(
            h->hdf5_file, "alpha_te", 0, 0, status);
    oskar_Mem* alpha_tm = oskar_hdf5_read_dataset(
            h->hdf5_file, "alpha_tm", 0, 0, status);
    oskar_Mem* coeffs_pola = oskar_hdf5_read_dataset(
            h->hdf5_file, "coeffs_pola", 0, 0, status);
    oskar_Mem* coeffs_polb = oskar_hdf5_read_dataset(
            h->hdf5_file, "coeffs_polb", 0, 0, status);
    if (*status)
    {
        oskar_mem_free(alpha_te, status);
        oskar_mem_free(alpha_tm, status);
        oskar_mem_free(coeffs_pola, status);
        oskar_mem_free(coeffs_polb, status);
        return;
    }
    h->alpha_te = alpha_te;
    if (oskar_mem_precision(alpha_te) != h->precision)
    {
        h->alpha_te = oskar_mem_convert_precision(
                alpha_te, h->precision, status);
        oskar_mem_free(alpha_te, status);
    }
    h->alpha_tm = alpha_tm;
    if (oskar_mem_precision(alpha_tm) != h->precision)
    {
        h->alpha_tm = oskar_mem_convert_precision(
                alpha_tm, h->precision, status);
        oskar_mem_free(alpha_tm, status);
    }
    h->coeffs_pola = coeffs_pola;
    if (oskar_mem_precision(coeffs_pola) != h->precision)
    {
        h->coeffs_pola = oskar_mem_convert_precision(
                coeffs_pola, h->precision, status);
        oskar_mem_free(coeffs_pola, status);
    }
    h->coeffs_polb = coeffs_polb;
    if (oskar_mem_precision(coeffs_polb) != h->precision)
    {
        h->coeffs_polb = oskar_mem_convert_precision(
                coeffs_polb, h->precision, status);
        oskar_mem_free(coeffs_polb, status);
    }
}
