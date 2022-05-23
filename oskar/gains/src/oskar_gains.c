/*
 * Copyright (c) 2020-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdlib.h>

#include "gains/oskar_gains.h"
#include "gains/private_gains.h"
#include "log/oskar_log.h"
#include "math/oskar_find_closest_match.h"

oskar_Gains* oskar_gains_create(int precision)
{
    oskar_Gains* h = (oskar_Gains*) calloc(1, sizeof(oskar_Gains));
    h->precision = precision;
    return h;
}

oskar_Gains* oskar_gains_create_copy(const oskar_Gains* other, int* status)
{
    oskar_Gains* h = (oskar_Gains*) calloc(1, sizeof(oskar_Gains));
    h->precision = other->precision;
    h->num_dims = other->num_dims;
    if (other->freqs)
    {
        h->freqs = oskar_mem_create_copy(other->freqs, OSKAR_CPU, status);
    }
    h->hdf5_file = other->hdf5_file;
    oskar_hdf5_ref_inc(h->hdf5_file);
    if (other->dims)
    {
        int i = 0;
        h->dims = (size_t*) calloc(h->num_dims, sizeof(size_t));
        for (i = 0; i < h->num_dims; ++i)
        {
            h->dims[i] = other->dims[i];
        }
    }
    return h;
}

int oskar_gains_defined(const oskar_Gains* h)
{
    return (h->hdf5_file != 0);
}

void oskar_gains_evaluate(const oskar_Gains* h, int time_index_sim,
        double frequency_hz, oskar_Mem* gains, int feed, int* status)
{
    oskar_Mem *temp_gains = 0, *temp_x = 0, *temp_y = 0;
    oskar_Mem *ptr_gains = 0, *ptr_x = 0, *ptr_y = 0;
    oskar_Mem *x = 0, *y = 0;
    int channel_index = 0;
    size_t i = 0;
    if (*status) return;

    /* Check data have been loaded. */
    if (!h->freqs || !h->hdf5_file)
    {
        oskar_log_error(0, "HDF5 file not opened.");
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;
        return;
    }

    /* Find the channel corresponding to this frequency. */
    channel_index = oskar_find_closest_match(frequency_hz, h->freqs, status);

    /* Bounds check. */
    if (time_index_sim >= (int) h->dims[0])
    {
        if ((int) h->dims[0] > 1)
        {
            oskar_log_warning(0,
                    "Time index %d out of range of HDF5 gain table (%d)",
                    time_index_sim, h->dims[0]);
        }
        time_index_sim = (int) h->dims[0] - 1;
    }
    if (channel_index >= (int) h->dims[1])
    {
        if ((int) h->dims[1] > 1)
        {
            oskar_log_warning(0,
                    "Channel index %d out of range of HDF5 gain table (%d)",
                    channel_index, h->dims[1]);
        }
        channel_index = (int) h->dims[1] - 1;
    }

    /* Get the dimensions to read. */
    const size_t num_antennas = h->dims[2];
    const size_t offsets[] = {time_index_sim, channel_index, 0};
    const size_t sizes[] = {1, 1, num_antennas};
    const int out_prec = oskar_mem_precision(gains);
    oskar_mem_ensure(gains, num_antennas, status);

    /* Check if requested gains are fully polarised. */
    if (oskar_mem_is_matrix(gains))
    {
        /* Read gains for X polarisation. */
        ptr_x = x = oskar_hdf5_read_hyperslab(h->hdf5_file, "gain_xpol",
                3, offsets, sizes, status);
        if (oskar_mem_precision(x) != out_prec)
        {
            ptr_x = temp_x = oskar_mem_convert_precision(x, out_prec, status);
        }

        /* Read gains for Y polarisation. */
        if (oskar_hdf5_dataset_exists(h->hdf5_file, "/gain_ypol"))
        {
            ptr_y = y = oskar_hdf5_read_hyperslab(h->hdf5_file, "gain_ypol",
                    3, offsets, sizes, status);
            if (oskar_mem_precision(y) != out_prec)
            {
                ptr_y = temp_y = oskar_mem_convert_precision(
                        y, out_prec, status);
            }
        }
        else
        {
            ptr_y = ptr_x;
        }

        /* Check output is writable by the CPU. */
        ptr_gains = gains;
        if (oskar_mem_location(gains) != OSKAR_CPU)
        {
            ptr_gains = temp_gains = oskar_mem_create(
                    oskar_mem_type(gains), OSKAR_CPU, num_antennas, status);
        }

        /* Write gains into diagonal matrices. */
        if (out_prec == OSKAR_DOUBLE)
        {
            double4c* out = 0;
            double2 zero = {0.0, 0.0};
            const double2* in_x = oskar_mem_double2_const(ptr_x, status);
            const double2* in_y = oskar_mem_double2_const(ptr_y, status);
            out = oskar_mem_double4c(ptr_gains, status);
            for (i = 0; i < num_antennas; ++i)
            {
                out[i].a = in_x[i];
                out[i].b = zero;
                out[i].c = zero;
                out[i].d = in_y[i];
            }
        }
        else
        {
            float4c* out = 0;
            float2 zero = {0.0f, 0.0f};
            const float2* in_x = oskar_mem_float2_const(ptr_x, status);
            const float2* in_y = oskar_mem_float2_const(ptr_y, status);
            out = oskar_mem_float4c(ptr_gains, status);
            for (i = 0; i < num_antennas; ++i)
            {
                out[i].a = in_x[i];
                out[i].b = zero;
                out[i].c = zero;
                out[i].d = in_y[i];
            }
        }

        /* Copy into output if necessary. */
        if (ptr_gains != gains)
        {
            oskar_mem_copy(gains, ptr_gains, status);
        }
    }
    else
    {
        /* Read gains only for specified polarisation. */
        const char* dataset = "gain_xpol";
        if (feed == 1 && oskar_hdf5_dataset_exists(h->hdf5_file, "/gain_ypol"))
        {
            dataset = "gain_ypol";
        }
        ptr_x = x = oskar_hdf5_read_hyperslab(h->hdf5_file, dataset,
                3, offsets, sizes, status);
        if (oskar_mem_precision(x) != out_prec)
        {
            ptr_x = temp_x = oskar_mem_convert_precision(x, out_prec, status);
        }
        oskar_mem_copy_contents(gains, ptr_x, 0, 0, num_antennas, status);
    }

    /* Free scratch memory. */
    oskar_mem_free(temp_gains, status);
    oskar_mem_free(temp_x, status);
    oskar_mem_free(temp_y, status);
    oskar_mem_free(x, status);
    oskar_mem_free(y, status);
}

void oskar_gains_free(oskar_Gains* h, int* status)
{
    if (!h) return;
    free(h->dims);
    oskar_mem_free(h->freqs, status);
    oskar_hdf5_close(h->hdf5_file);
    free(h);
}

void oskar_gains_open_hdf5(oskar_Gains* h, const char* path, int* status)
{
    if (*status) return;
    h->hdf5_file = oskar_hdf5_open(path, status);

    /* Load the frequency channel map. */
    oskar_mem_free(h->freqs, status);
    h->freqs = oskar_hdf5_read_dataset(h->hdf5_file, "freq (Hz)", 0, 0, status);

    /* Get the size of the gain table. */
    oskar_hdf5_read_dataset_dims(h->hdf5_file, "gain_xpol",
            &h->num_dims, &h->dims, status);

    /* Check the array is 3-dimensional. */
    if (h->num_dims != 3)
    {
        oskar_log_error(0, "HDF5 gain tables must be 3-dimensional.");
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Check the frequency dimensions match. */
    if (oskar_mem_length(h->freqs) != h->dims[1])
    {
        oskar_log_error(0,
                "Inconsistent frequency dimensions in HDF5 gain table.");
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }
}
