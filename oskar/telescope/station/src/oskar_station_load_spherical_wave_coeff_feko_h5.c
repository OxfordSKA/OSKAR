/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "math/oskar_cmath.h"
#include "log/oskar_log.h"
#include "telescope/station/oskar_station.h"
#include "telescope/station/element/private_element.h"
#include "utility/oskar_hdf5.h"

#ifdef __cplusplus
extern "C" {
#endif

#if __STDC_VERSION__ >= 199901L
#define SNPRINTF(BUF, SIZE, FMT, ...) snprintf(BUF, SIZE, FMT, __VA_ARGS__);
#else
#define SNPRINTF(BUF, SIZE, FMT, ...) sprintf(BUF, FMT, __VA_ARGS__);
#endif


/* Convert dataset dimensions to maximum order of spherical wave. */
static int dims_to_l_max(const char* dataset, const size_t* dims, int* status)
{
    const size_t num_coeffs = dims[1] / 2; /* Factor 2 for TE & TM. */
    const int l_max = (int) (sqrt(1 + num_coeffs) - 1 + 0.5);
    const int expected_length = 2 * (l_max * l_max - 1 + (2 * l_max + 1));
    if (dims[1] != (size_t) expected_length)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        oskar_log_error(
                0, "Dataset '%s' does not have the expected size for "
                "l_max=%d (actual=%d, expected=%d)",
                dataset, l_max, (int) dims[1], expected_length
        );
    }
    return *status ? 0 : l_max;
}


/*
 * Split out data into complex arrays for TE and TM.
 * The dataset contains interleaved TE, TM values, and we have all the
 * amplitudes first, then all the phases (in degrees).
 *   AMP_TE,   AMP_TM,   AMP_TE,   AMP_TM   ...
 *   PHASE_TE, PHASE_TM, PHASE_TE, PHASE_TM ...
 */
static void extract_te_tm_from_dataset(
        const oskar_Mem* dataset,
        int dataset_dim,
        int l_max_alloc,
        int l_max_dataset,
        oskar_Mem** te,
        oskar_Mem** tm,
        int* status
)
{
    int j = 0, k = 0;
    if (*status) return;
    const double deg2rad = M_PI / 180.0;
    const size_t num_coeff = (l_max_alloc + 1) * (l_max_alloc + 1) - 1;
    *te = oskar_mem_create(OSKAR_DOUBLE_COMPLEX, OSKAR_CPU, num_coeff, status);
    *tm = oskar_mem_create(OSKAR_DOUBLE_COMPLEX, OSKAR_CPU, num_coeff, status);
    oskar_mem_clear_contents(*te, status);
    oskar_mem_clear_contents(*tm, status);
    double2* te_ = oskar_mem_double2(*te, status);
    double2* tm_ = oskar_mem_double2(*tm, status);
    if (oskar_mem_precision(dataset) == OSKAR_DOUBLE)
    {
        const double* data_ = oskar_mem_double_const(dataset, status);
        for (j = 1; j <= l_max_alloc && j <= l_max_dataset; ++j)
        {
            const int length = 2 * j + 1;
            const int index_start = j * j - 1;
            for (k = 0; k < length; ++k)
            {
                const int idx_amp = 2 * (index_start + k);
                const int idx_pha = 2 * (index_start + k) + dataset_dim;
                const double a_te = data_[idx_amp + 0];
                const double a_tm = data_[idx_amp + 1];
                const double p_te = data_[idx_pha + 0] * deg2rad;
                const double p_tm = data_[idx_pha + 1] * deg2rad;
                const double2 e = {a_te * cos(p_te), a_te * sin(p_te)};
                const double2 m = {a_tm * cos(p_tm), a_tm * sin(p_tm)};
                te_[index_start + k] = e;
                tm_[index_start + k] = m;
            }
        }
    }
    else
    {
        const float* data_ = oskar_mem_float_const(dataset, status);
        for (j = 1; j <= l_max_alloc && j <= l_max_dataset; ++j)
        {
            const int length = 2 * j + 1;
            const int index_start = j * j - 1;
            for (k = 0; k < length; ++k)
            {
                const int idx_amp = 2 * (index_start + k);
                const int idx_pha = 2 * (index_start + k) + dataset_dim;
                const double a_te = data_[idx_amp + 0];
                const double a_tm = data_[idx_amp + 1];
                const double p_te = data_[idx_pha + 0] * deg2rad;
                const double p_tm = data_[idx_pha + 1] * deg2rad;
                const double2 e = {a_te * cos(p_te), a_te * sin(p_te)};
                const double2 m = {a_tm * cos(p_tm), a_tm * sin(p_tm)};
                te_[index_start + k] = e;
                tm_[index_start + k] = m;
            }
        }
    }
}


/* Scan all datasets in the file to find number of antennas and frequencies. */
static double* scan_datasets(
        oskar_HDF5* file,
        int* num_antennas,
        int* num_freqs,
        const int* status
)
{
    int i_dataset = 0;
    double* freqs_hz = 0;
    if (*status) return 0;
    const int num_datasets = oskar_hdf5_num_datasets(file);
    for (i_dataset = 0; i_dataset < num_datasets; ++i_dataset)
    {
        /* Parse the dataset name. Element indices here start at 1, not 0. */
        char pol = ' ';
        int index = 0;
        double freq_hz = 0.0;
        const char* dataset_name = oskar_hdf5_dataset_name(file, i_dataset);
        const char* start = strpbrk(dataset_name, "XY");
        if (start && sscanf(start, "%c%d_%lf", &pol, &index, &freq_hz) == 3)
        {
            int i_freq = 0;
            for (i_freq = 0; i_freq < *num_freqs; ++i_freq)
            {
                if (fabs(freqs_hz[i_freq] - freq_hz) <= freq_hz * DBL_EPSILON)
                    break;
            }
            if (i_freq >= *num_freqs)
            {
                (*num_freqs)++;
                freqs_hz = (double*) realloc(
                        freqs_hz, *num_freqs * sizeof(double)
                );
                freqs_hz[i_freq] = freq_hz;
            }
            if (index > *num_antennas) *num_antennas = index;
        }
    }
    return freqs_hz;
}


/* Stores data in the element model. */
static void store_element_data(
        oskar_Element* element,
        double freq_hz,
        int l_max,
        oskar_Mem* te[2],
        oskar_Mem* tm[2],
        int* status
)
{
    int i = 0, f = 0;
    oskar_Mem* t_te[] = {0, 0};
    oskar_Mem* t_tm[] = {0, 0};
    oskar_Mem* p_te[] = {te[0], te[1]};
    oskar_Mem* p_tm[] = {tm[0], tm[1]};
    if (*status) return;

    /* Check if this frequency has already been set in the element data,
     * and get its index into 'f' if so. */
    for (f = 0; f < element->num_freq; ++f)
    {
        if (fabs(element->freqs_hz[f] - freq_hz) <= freq_hz * DBL_EPSILON)
        {
            break;
        }
    }

    /* Otherwise expand element arrays for a new frequency. */
    if (f >= element->num_freq)
    {
        f = element->num_freq;
        oskar_element_resize_freq_data(element, f + 1, status);
        element->freqs_hz[f] = freq_hz;
    }

    /* Store the meta-data. */
    element->l_max[f] = l_max;
    element->common_phi_coords[f] = 1;

    /* Ensure there is space to store the coefficients. */
    const int num_coeff = (l_max + 1) * (l_max + 1) - 1;
    if (!element->sph_wave_feko[f])
    {
        element->sph_wave_feko[f] = oskar_mem_create(
                element->precision | OSKAR_COMPLEX | OSKAR_MATRIX,
                OSKAR_CPU, (size_t) num_coeff, status
        );
    }
    oskar_mem_ensure(element->sph_wave_feko[f], (size_t) num_coeff, status);

    /* Convert precision if required. */
    if (element->precision == OSKAR_SINGLE)
    {
        for (i = 0; i < 2; ++i)
        {
            t_te[i] = oskar_mem_convert_precision(te[i], OSKAR_SINGLE, status);
            t_tm[i] = oskar_mem_convert_precision(tm[i], OSKAR_SINGLE, status);
            p_te[i] = t_te[i];
            p_tm[i] = t_tm[i];
        }
    }

    /* Store the coefficients. */
    if (element->precision == OSKAR_SINGLE)
    {
        float4c* sw = oskar_mem_float4c(element->sph_wave_feko[f], status);
        const float2* x_te = oskar_mem_float2_const(p_te[0], status);
        const float2* x_tm = oskar_mem_float2_const(p_tm[0], status);
        const float2* y_te = oskar_mem_float2_const(p_te[1], status);
        const float2* y_tm = oskar_mem_float2_const(p_tm[1], status);
        for (i = 0; i < num_coeff; ++i)
        {
            sw[i].a = x_te[i];
            sw[i].b = x_tm[i];
            sw[i].c = y_te[i];
            sw[i].d = y_tm[i];
        }
    }
    else
    {
        double4c* sw = oskar_mem_double4c(element->sph_wave_feko[f], status);
        const double2* x_te = oskar_mem_double2_const(p_te[0], status);
        const double2* x_tm = oskar_mem_double2_const(p_tm[0], status);
        const double2* y_te = oskar_mem_double2_const(p_te[1], status);
        const double2* y_tm = oskar_mem_double2_const(p_tm[1], status);
        for (i = 0; i < num_coeff; ++i)
        {
            sw[i].a = x_te[i];
            sw[i].b = x_tm[i];
            sw[i].c = y_te[i];
            sw[i].d = y_tm[i];
        }
    }
    for (i = 0; i < 2; ++i)
    {
        oskar_mem_free(t_te[i], status);
        oskar_mem_free(t_tm[i], status);
    }
}


void oskar_station_load_spherical_wave_coeff_feko_h5(
        oskar_Station* station,
        const char* filename,
        int max_order,
        int* status
)
{
    oskar_Mem* dataset[] = {0, 0};
    if (*status || !station) return;

    /* Scan the file to find the number of antennas and frequencies. */
    int i_antenna = 0, i_freq = 0, i_pol = 0, num_antennas = 0, num_freqs = 0;
    oskar_HDF5* file = oskar_hdf5_open(filename, 'r', status);
    double* freqs_hz = scan_datasets(file, &num_antennas, &num_freqs, status);

    /* Ensure there are enough element types in the station. */
    if (oskar_station_num_element_types(station) < num_antennas)
    {
        oskar_station_resize_element_types(station, num_antennas, status);
    }

    /* Loop over antennas. */
    for (i_antenna = 0; i_antenna < num_antennas; ++i_antenna)
    {
        if (*status) break;

        /* Get a pointer to the element data for this antenna. */
        oskar_Element* element = oskar_station_element(station, i_antenna);

        /* Loop over frequencies. */
        for (i_freq = 0; i_freq < num_freqs; ++i_freq)
        {
            oskar_Mem *te[] = {0, 0}, *tm[] = {0, 0};
            size_t* dims[] = {0, 0};
            int num_dims[] = {0, 0};
            int l_max[] = {0, 0};
            int l_max_actual = 0;
            char dataset_name[2][32];
            if (*status) break;

            /* Loop over polarisations. */
            for (i_pol = 0; i_pol < 2; ++i_pol)
            {
                /* Load the dataset for each polarisation. */
                static const char pol_name[2] = {'X', 'Y'};
                (void) SNPRINTF(
                        dataset_name[i_pol], sizeof(dataset_name[i_pol]),
                        "%c%d_%.0f",
                        pol_name[i_pol], i_antenna + 1, freqs_hz[i_freq]
                );
                oskar_mem_free(dataset[i_pol], status);
                dataset[i_pol] = oskar_hdf5_read_dataset(
                        file, 0, dataset_name[i_pol],
                        &num_dims[i_pol], &dims[i_pol], status
                );
                if (num_dims[i_pol] != 2) continue; /* Ignore if not 2D. */

                /* Get l_max for each polarisation. */
                l_max[i_pol] = dims_to_l_max(
                        dataset_name[i_pol], dims[i_pol], status
                );
            }

            /*
             * Find a single value of l_max to use for both polarisations.
             * This needs to be big enough to support the larger l_max,
             * and coefficients for the other polarisation should be padded
             * with zeros if required.
             */
            l_max_actual = l_max[0] > l_max[1] ? l_max[0] : l_max[1];
            if (max_order > 0)
            {
                /* Clamp to supplied maximum order if valid. */
                l_max_actual = (
                        max_order < l_max_actual ? max_order : l_max_actual
                );
            }

            /* Extract TE and TM coefficients up to l_max_actual. */
            for (i_pol = 0; i_pol < 2; ++i_pol)
            {
                if (*status) break;
                extract_te_tm_from_dataset(
                        dataset[i_pol], (int) dims[i_pol][1], l_max_actual,
                        l_max[i_pol], &te[i_pol], &tm[i_pol], status
                );
            }

            /* Write data to element model. */
            store_element_data(
                    element, freqs_hz[i_freq], l_max_actual, te, tm, status
            );
            for (i_pol = 0; i_pol < 2; ++i_pol)
            {
                oskar_mem_free(te[i_pol], status);
                oskar_mem_free(tm[i_pol], status);
                free(dims[i_pol]);
            }
        }
    }
    oskar_hdf5_close(file);
    oskar_mem_free(dataset[0], status);
    oskar_mem_free(dataset[1], status);
    free(freqs_hz);
}

#ifdef __cplusplus
}
#endif
