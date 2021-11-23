/*
 * Copyright (c) 2019-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifdef OSKAR_HAVE_CUDA
#include <cufft.h>
#endif

#include "log/oskar_log.h"
#include "math/oskar_fft.h"
#include "math/oskar_fftpack_cfft.h"
#include "math/oskar_fftpack_cfft_f.h"

#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_FFT
{
    size_t num_cells_total;
    oskar_Mem *fftpack_work, *fftpack_wsave;
    int precision, location, num_dim, dim_size, ensure_consistent_norm;
#ifdef OSKAR_HAVE_CUDA
    cufftHandle cufft_plan;
#endif
};

#ifdef OSKAR_HAVE_CUDA
static void print_cufft_error(cufftResult code)
{
    switch (code)
    {
    case CUFFT_INVALID_PLAN:
        oskar_log_error(0, "Invalid CUFFT plan.");
        break;
    case CUFFT_ALLOC_FAILED:
        oskar_log_error(0, "CUFFT memory allocation failed.");
        break;
    case CUFFT_INTERNAL_ERROR:
        oskar_log_error(0, "CUFFT internal error.");
        break;
    case CUFFT_EXEC_FAILED:
        oskar_log_error(0, "CUFFT exec failed.");
        break;
    case CUFFT_SETUP_FAILED:
        oskar_log_error(0, "CUFFT setup failed.");
        break;
    case CUFFT_INVALID_SIZE:
        oskar_log_error(0, "CUFFT invalid size.");
        break;
    case CUFFT_INVALID_VALUE:
        oskar_log_error(0, "CUFFT invalid value.");
        break;
    case CUFFT_UNALIGNED_DATA:
        oskar_log_error(0, "CUFFT unaligned data.");
        break;
    case CUFFT_NO_WORKSPACE:
        oskar_log_error(0, "CUFFT no workspace.");
        break;
    default:
        oskar_log_error(0, "CUFFT error, code %d", code);
        break;
    }
}
#endif

oskar_FFT* oskar_fft_create(int precision, int location, int num_dim,
        int dim_size, int batch_size_1d, int* status)
{
    int i = 0;
    oskar_FFT* h = (oskar_FFT*) calloc(1, sizeof(oskar_FFT));
#ifndef OSKAR_HAVE_CUDA
    if (location == OSKAR_GPU) location = OSKAR_CPU;
#endif
#ifndef OSKAR_HAVE_OPENCL
    if (location & OSKAR_CL) location = OSKAR_CPU;
#endif
    h->precision = precision;
    h->location = location;
    h->num_dim = num_dim;
    h->dim_size = dim_size;
    h->ensure_consistent_norm = 1;
    h->num_cells_total = (size_t) dim_size;
    for (i = 1; i < num_dim; ++i) h->num_cells_total *= (size_t) dim_size;
    if (location == OSKAR_CPU || (location & OSKAR_CL))
    {
        int len = 4 * dim_size +
                2 * (int)(log((double)dim_size) / log(2.0)) + 8;
        if (location & OSKAR_CL)
        {
            h->location = OSKAR_CPU;
            oskar_log_warning(0,
                    "OpenCL FFT not implemented; using CPU version instead.");
        }
        h->fftpack_wsave = oskar_mem_create(precision, OSKAR_CPU, len, status);
        if (num_dim == 1)
        {
            (void) batch_size_1d;
            *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
        }
        else if (num_dim == 2)
        {
            if (precision == OSKAR_DOUBLE)
            {
                oskar_fftpack_cfft2i(dim_size, dim_size,
                        oskar_mem_double(h->fftpack_wsave, status));
            }
            else
            {
                oskar_fftpack_cfft2i_f(dim_size, dim_size,
                        oskar_mem_float(h->fftpack_wsave, status));
            }
        }
        else
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
        }
        h->fftpack_work = oskar_mem_create(precision, OSKAR_CPU,
                2 * h->num_cells_total, status);
    }
    else if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        cufftResult cufft_error_code = CUFFT_SUCCESS;
        if (num_dim == 1)
        {
            cufft_error_code = cufftPlan1d(&h->cufft_plan, dim_size,
                    ((precision == OSKAR_DOUBLE) ? CUFFT_Z2Z : CUFFT_C2C),
                    batch_size_1d);
        }
        else if (num_dim == 2)
        {
            cufft_error_code = cufftPlan2d(&h->cufft_plan, dim_size, dim_size,
                    ((precision == OSKAR_DOUBLE) ? CUFFT_Z2Z : CUFFT_C2C));
        }
        else
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
        }
        if (cufft_error_code != CUFFT_SUCCESS)
        {
            *status = OSKAR_ERR_FFT_FAILED;
            print_cufft_error(cufft_error_code);
        }
#endif
    }
    else
    {
        *status = OSKAR_ERR_BAD_LOCATION;
    }
    return h;
}

void oskar_fft_exec(oskar_FFT* h, oskar_Mem* data, int* status)
{
    oskar_Mem *data_copy = 0, *data_ptr = data;
    if (*status) return;
    if (oskar_mem_location(data) != h->location)
    {
        data_copy = oskar_mem_create_copy(data, h->location, status);
        data_ptr = data_copy;
    }
    if (h->location == OSKAR_CPU)
    {
        if (h->num_dim == 1)
        {
            *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
        }
        else if (h->num_dim == 2)
        {
            if (h->precision == OSKAR_DOUBLE)
            {
                oskar_fftpack_cfft2f(h->dim_size, h->dim_size, h->dim_size,
                        oskar_mem_double(data_ptr, status),
                        oskar_mem_double(h->fftpack_wsave, status),
                        oskar_mem_double(h->fftpack_work, status));
            }
            else
            {
                oskar_fftpack_cfft2f_f(h->dim_size, h->dim_size, h->dim_size,
                        oskar_mem_float(data_ptr, status),
                        oskar_mem_float(h->fftpack_wsave, status),
                        oskar_mem_float(h->fftpack_work, status));
            }

            /* This step not needed for W-kernel generation, so turn it off. */
            if (h->ensure_consistent_norm)
            {
                oskar_mem_scale_real(data_ptr, (double)h->num_cells_total,
                        0, h->num_cells_total, status);
            }
        }
    }
    else if (h->location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        cufftResult cufft_error_code = CUFFT_SUCCESS;
        if (h->precision == OSKAR_DOUBLE)
        {
            cufft_error_code = cufftExecZ2Z(h->cufft_plan,
                    (cufftDoubleComplex*) oskar_mem_void(data_ptr),
                    (cufftDoubleComplex*) oskar_mem_void(data_ptr),
                    CUFFT_FORWARD);
        }
        else
        {
            cufft_error_code = cufftExecC2C(h->cufft_plan,
                    (cufftComplex*) oskar_mem_void(data_ptr),
                    (cufftComplex*) oskar_mem_void(data_ptr),
                    CUFFT_FORWARD);
        }
        if (cufft_error_code != CUFFT_SUCCESS)
        {
            *status = OSKAR_ERR_FFT_FAILED;
            print_cufft_error(cufft_error_code);
        }
#endif
    }
    else
    {
        *status = OSKAR_ERR_BAD_LOCATION;
    }
    if (oskar_mem_location(data) != h->location)
    {
        oskar_mem_copy(data, data_ptr, status);
    }
    oskar_mem_free(data_copy, status);
}

void oskar_fft_free(oskar_FFT* h)
{
    int status = 0;
    if (!h) return;
    oskar_mem_free(h->fftpack_work, &status);
    oskar_mem_free(h->fftpack_wsave, &status);
#ifdef OSKAR_HAVE_CUDA
    if (h->location == OSKAR_GPU)
    {
        cufftDestroy(h->cufft_plan);
    }
#endif
    free(h);
}

void oskar_fft_set_ensure_consistent_norm(oskar_FFT* h, int value)
{
    h->ensure_consistent_norm = value;
}

#ifdef __cplusplus
}
#endif
